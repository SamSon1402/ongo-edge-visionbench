"""ONNX Runtime adapter.

Real implementation. Lazy-imports `onnxruntime` so the package works on
machines that don't have it (e.g. CI for the core module).

Supports CPU, CUDA, and TensorRT execution providers — selected by target.
For Jetson we hand-tune session options (intra/inter-op threads, graph
optimization level).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ongoedge.core import (
    BoundingBox,
    Detection,
    EngineBuildError,
    Frame,
    LatencySample,
    Precision,
    Target,
)
from ongoedge.runtimes.base import RuntimeBase, timed

if TYPE_CHECKING:
    import onnxruntime as ort  # pragma: no cover


class OnnxRuntime(RuntimeBase):
    """Adapter from a YOLO-style ONNX detector to the ModelRuntime protocol."""

    # Execution provider lookup by target. Tuples preserve preference order.
    _PROVIDER_PRIORITY: dict[Target, tuple[str, ...]] = {
        Target.HOST_CPU: ("CPUExecutionProvider",),
        Target.JETSON_ORIN_NANO: ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        Target.JETSON_ORIN_NX: ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        Target.RPI5: ("CPUExecutionProvider",),
        Target.CORAL_EDGETPU: ("CPUExecutionProvider",),  # coral uses tflite path
        Target.RK3588_NPU: ("CPUExecutionProvider",),     # rk3588 uses tflite path
    }

    def __init__(
        self,
        *,
        model_path: Path,
        model_id: str,
        target: Target,
        precision: Precision,
        input_name: str = "images",
        score_threshold: float = 0.25,
        labels: list[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._model_id = model_id
        self._target = target
        self._precision = precision
        self._input_name = input_name
        self._score_threshold = score_threshold
        self._labels = labels or ["object"]
        self._session: ort.InferenceSession | None = None
        self._init_session()

    # ── ModelRuntime protocol ──────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def target(self) -> Target:
        return self._target

    @property
    def precision(self) -> Precision:
        return self._precision

    def warmup(self, n: int) -> None:
        assert self._session is not None
        dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
        for _ in range(n):
            self._session.run(None, {self._input_name: dummy})

    def infer(self, frame: Frame) -> tuple[list[Detection], LatencySample]:
        assert self._session is not None

        with timed() as t_pre:
            tensor = self._preprocess(frame)

        with timed() as t_inf:
            raw = self._session.run(None, {self._input_name: tensor})

        with timed() as t_post:
            detections = self._postprocess(raw, frame.width, frame.height)

        return detections, LatencySample(
            preprocess_ms=t_pre.elapsed_ms,
            inference_ms=t_inf.elapsed_ms,
            postprocess_ms=t_post.elapsed_ms,
        )

    def close(self) -> None:
        # onnxruntime sessions release on GC; we drop the reference to make it explicit.
        self._session = None

    # ── internals ──────────────────────────────────────────────────

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise EngineBuildError(
                "onnxruntime not installed. Install with `pip install onnxruntime` "
                "or for GPU `pip install onnxruntime-gpu`."
            ) from exc

        if not self._model_path.exists():
            raise EngineBuildError(f"model not found: {self._model_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # On Jetson Orin Nano, 4 cores is the sweet spot. On host we let ORT pick.
        if self._target == Target.JETSON_ORIN_NANO:
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1

        providers: list[str | tuple[str, dict[str, Any]]] = []
        for name in self._PROVIDER_PRIORITY[self._target]:
            if name == "TensorrtExecutionProvider" and self._precision in {
                Precision.FP16,
                Precision.INT8,
            }:
                providers.append(
                    (
                        name,
                        {
                            "trt_fp16_enable": self._precision == Precision.FP16,
                            "trt_int8_enable": self._precision == Precision.INT8,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "./.trt_cache",
                        },
                    )
                )
            else:
                providers.append(name)

        try:
            self._session = ort.InferenceSession(
                str(self._model_path), sess_options=opts, providers=providers
            )
        except Exception as exc:
            raise EngineBuildError(
                f"failed to load {self._model_path} with providers={providers}: {exc}"
            ) from exc

    @staticmethod
    def _preprocess(frame: Frame) -> np.ndarray:
        """Resize-keep-aspect → letterbox → CHW float32 normalized to 0..1.

        We assume the model wants 320×320 to keep this minimal — the production
        version reads input shape from the session metadata.
        """
        target = 320
        h, w = frame.height, frame.width
        scale = target / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Naive resize using array slicing (placeholder; production = opencv).
        # The point here is the interface, not the bilinear quality.
        pixels = frame.pixels.astype(np.float32) / 255.0
        # We don't have cv2 in deps for the skeleton; nearest-resample.
        ys = (np.arange(new_h) * (h / new_h)).astype(np.int32)
        xs = (np.arange(new_w) * (w / new_w)).astype(np.int32)
        resized = pixels[ys][:, xs]

        canvas = np.zeros((target, target, 3), dtype=np.float32)
        canvas[:new_h, :new_w] = resized

        # HWC → CHW → NCHW
        return canvas.transpose(2, 0, 1)[None, ...]

    def _postprocess(
        self, raw: list[np.ndarray], orig_w: int, orig_h: int
    ) -> list[Detection]:
        """Parse YOLO-style output: (N, 6) = [x1, y1, x2, y2, score, cls].

        Real impl would handle anchors / decoders / NMS. This is the API
        surface, hardware-tuned NMS lives in a separate module.
        """
        if not raw:
            return []
        preds = raw[0]
        if preds.ndim == 3:
            preds = preds[0]  # drop batch
        if preds.shape[-1] < 6:
            return []

        out: list[Detection] = []
        for row in preds:
            score = float(row[4])
            if score < self._score_threshold:
                continue
            cls = int(row[5])
            label = self._labels[cls] if cls < len(self._labels) else f"cls_{cls}"
            out.append(
                Detection(
                    label=label,
                    confidence=score,
                    box=BoundingBox(
                        x1=float(row[0]),
                        y1=float(row[1]),
                        x2=float(row[2]),
                        y2=float(row[3]),
                    ),
                )
            )
        return out
