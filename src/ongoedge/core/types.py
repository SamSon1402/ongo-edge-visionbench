"""Domain types.

Everything is either a frozen pydantic model or a plain Enum so that
benchmark runs are reproducible, serializable, and hashable. No
mutable shared state escapes a module boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── primitives ──────────────────────────────────────────────────────


class Precision(StrEnum):
    """Inference precision. Order matters: smaller = faster, less accurate."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bytes_per_param(self) -> float:
        return {self.FP32: 4.0, self.FP16: 2.0, self.INT8: 1.0, self.INT4: 0.5}[self]


class Target(StrEnum):
    """Edge target device. Add new SoCs here; runtimes dispatch on this."""

    HOST_CPU = "host-cpu"
    JETSON_ORIN_NANO = "jetson-orin-nano"
    JETSON_ORIN_NX = "jetson-orin-nx"
    RPI5 = "rpi5"
    CORAL_EDGETPU = "coral-edgetpu"
    RK3588_NPU = "rk3588-npu"

    @property
    def is_arm(self) -> bool:
        return self in {self.JETSON_ORIN_NANO, self.JETSON_ORIN_NX, self.RPI5, self.RK3588_NPU}


# ── data ────────────────────────────────────────────────────────────


class Frame(BaseModel):
    """A camera frame ready for inference. Backed by numpy under the hood."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixels: np.ndarray  # (H, W, 3) uint8, RGB
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        h, w, c = self.pixels.shape
        if (h, w) != (self.height, self.width) or c != 3:
            raise ValueError(
                f"pixel shape {self.pixels.shape} does not match declared "
                f"({self.height}, {self.width}, 3)"
            )
        if self.pixels.dtype != np.uint8:
            raise ValueError(f"pixels must be uint8, got {self.pixels.dtype}")
        return self


class BoundingBox(BaseModel):
    """Pixel-space axis-aligned box. xyxy convention, half-open on max."""

    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: {self}")
        return self

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: BoundingBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    """One object detected in one frame."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    box: BoundingBox
    track_id: int | None = None  # populated by tracker, optional


# ── timing ──────────────────────────────────────────────────────────


class LatencySample(BaseModel):
    """One frame's full pipeline timing, in milliseconds.

    We carry every stage because Ongo's UX is set by the bad tail and
    we need to know *which* stage produced the spike (preprocess?
    inference? postprocess?), not just the total.
    """

    model_config = ConfigDict(frozen=True)

    preprocess_ms: float = Field(ge=0)
    inference_ms: float = Field(ge=0)
    postprocess_ms: float = Field(ge=0)

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class TelemetrySnapshot(BaseModel):
    """A single point sample of device telemetry."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    ram_mb: float = Field(ge=0)
    power_w: float = Field(ge=0)
    soc_temp_c: float
    gpu_util_pct: float | None = Field(default=None, ge=0, le=100)
    cpu_util_pct: float | None = Field(default=None, ge=0, le=100)


# ── benchmark plan & result ─────────────────────────────────────────


class BenchPlan(BaseModel):
    """A reproducible benchmark recipe. Hash this for cache keys."""

    model_config = ConfigDict(frozen=True)

    target: Target
    model_id: str  # e.g. "yolo11n"
    precision: Precision
    input_h: int = Field(gt=0)
    input_w: int = Field(gt=0)
    num_frames: int = Field(gt=0, le=10_000)
    warmup_frames: int = Field(ge=0, default=10)
    seed: int = 42

    @property
    def plan_id(self) -> str:
        """Stable short id for filenames and dashboards."""
        return f"{self.target}_{self.model_id}_{self.precision}_{self.input_w}x{self.input_h}"


class BenchResult(BaseModel):
    """The outcome of executing a BenchPlan."""

    model_config = ConfigDict(frozen=True)

    plan: BenchPlan
    started_at: datetime
    finished_at: datetime
    latencies: list[LatencySample]
    telemetry: list[TelemetrySnapshot]
    mean_ap_50: float | None = Field(default=None, ge=0, le=1)

    # ── statistics (pre-computed so consumers don't all redo it) ──

    @property
    def p50_ms(self) -> float:
        return _quantile([s.total_ms for s in self.latencies], 0.5)

    @property
    def p99_ms(self) -> float:
        return _quantile([s.total_ms for s in self.latencies], 0.99)

    @property
    def fps(self) -> float:
        if not self.latencies:
            return 0.0
        return 1000.0 / max(self.p50_ms, 1e-6)

    @property
    def peak_ram_mb(self) -> float:
        return max((t.ram_mb for t in self.telemetry), default=0.0)

    @property
    def mean_power_w(self) -> float:
        if not self.telemetry:
            return 0.0
        return sum(t.power_w for t in self.telemetry) / len(self.telemetry)

    def frontier_score(self, lat_budget_ms: float = 33.0, power_budget_w: float = 8.0) -> float:
        """Single scalar for sorting the Pareto frontier.

        Higher is better. Penalizes blowing through latency / power budget.
        Accuracy is a multiplier so a tiny fast model with garbage mAP
        still loses to a slightly slower one with usable accuracy.
        """
        lat_score = min(1.0, lat_budget_ms / max(self.p99_ms, 1e-6))
        power_score = min(1.0, power_budget_w / max(self.mean_power_w, 1e-6))
        acc_score = self.mean_ap_50 if self.mean_ap_50 is not None else 0.5
        return lat_score * power_score * acc_score


# ── helpers ─────────────────────────────────────────────────────────


def _quantile(values: list[float], q: float) -> float:
    """Plain quantile (linear interp). We don't pull numpy for this hot path."""
    if not values:
        return 0.0
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0,1], got {q}")
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac
