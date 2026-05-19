"""Model registry.

A model entry knows:
  * Its canonical id (used in BenchPlan)
  * Its weights URL / local path
  * Its expected input shape & label set

Production: load this from configs/models.yaml. For the skeleton it's inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    weights_path: Path
    input_h: int
    input_w: int
    labels: tuple[str, ...]
    description: str


REGISTRY: dict[str, ModelSpec] = {
    "yolo11n": ModelSpec(
        model_id="yolo11n",
        weights_path=Path("weights/yolo11n.onnx"),
        input_h=320,
        input_w=320,
        labels=("person", "hand", "object"),
        description="YOLO11 nano — production default for Ongo perception.",
    ),
    "yolov8n": ModelSpec(
        model_id="yolov8n",
        weights_path=Path("weights/yolov8n.onnx"),
        input_h=320,
        input_w=320,
        labels=("person", "hand", "object"),
        description="YOLOv8 nano — fallback when YOLO11 won't quantize cleanly.",
    ),
    "mobilevit-xs": ModelSpec(
        model_id="mobilevit-xs",
        weights_path=Path("weights/mobilevit_xs.onnx"),
        input_h=256,
        input_w=256,
        labels=("person", "hand", "object"),
        description="MobileViT XS — transformer baseline.",
    ),
    "efficientvit-b0": ModelSpec(
        model_id="efficientvit-b0",
        weights_path=Path("weights/efficientvit_b0.onnx"),
        input_h=256,
        input_w=256,
        labels=("person", "hand", "object"),
        description="EfficientViT B0 — best accuracy in the small-ViT class.",
    ),
    "nanodet-plus": ModelSpec(
        model_id="nanodet-plus",
        weights_path=Path("weights/nanodet_plus.onnx"),
        input_h=320,
        input_w=320,
        labels=("person", "hand", "object"),
        description="NanoDet-Plus — lowest-latency anchor-free detector.",
    ),
}


def get(model_id: str) -> ModelSpec:
    try:
        return REGISTRY[model_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown model_id={model_id!r}. Available: {sorted(REGISTRY)}"
        ) from exc


__all__ = ["ModelSpec", "REGISTRY", "get"]
