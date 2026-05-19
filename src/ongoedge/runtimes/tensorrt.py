"""TensorRT runtime adapter.

Skeleton. The full builder (parser → engine → context → bindings) lives
in `engines.py` once we have target silicon. Keeping it scaffolded here
so the rest of the system already routes correctly.

Why scaffold and not full-impl: TRT engine builds are device-specific
(SM version, driver, plan files don't move between Jetsons). We need
the hardware in hand to write the real serialization path. The seams
below are exactly where that code drops in.
"""

from __future__ import annotations

from pathlib import Path

from ongoedge.core import (
    Detection,
    EngineBuildError,
    Frame,
    LatencySample,
    Precision,
    Target,
)
from ongoedge.runtimes.base import RuntimeBase


class TensorRTRuntime(RuntimeBase):
    """Loads a serialized .plan engine and runs inference via PyCUDA streams.

    TODO(jetson-in-hand):
      - Implement Builder.from_onnx() in engines.py (uses INetworkDefinition).
      - Pin host-side input buffer; reuse across calls.
      - Stream-async H2D, infer, D2H; one CUDA event per stage for timing.
      - INT8 path needs a Calibrator (see core/protocols.py).
    """

    SUPPORTED_TARGETS = {Target.JETSON_ORIN_NANO, Target.JETSON_ORIN_NX}

    def __init__(
        self,
        *,
        engine_path: Path,
        model_id: str,
        target: Target,
        precision: Precision,
    ) -> None:
        if target not in self.SUPPORTED_TARGETS:
            raise EngineBuildError(
                f"TensorRTRuntime: target {target} not in {self.SUPPORTED_TARGETS}"
            )
        self._engine_path = Path(engine_path)
        self._model_id = model_id
        self._target = target
        self._precision = precision
        # Real fields, populated by _load_engine() once implemented:
        # self._engine, self._context, self._stream, self._bindings
        raise NotImplementedError(
            "TensorRTRuntime is scaffolded — real engine loading lands once "
            "we have Jetson hardware in hand. See module docstring for the plan."
        )

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
        raise NotImplementedError

    def infer(self, frame: Frame) -> tuple[list[Detection], LatencySample]:
        raise NotImplementedError

    def close(self) -> None:
        pass
