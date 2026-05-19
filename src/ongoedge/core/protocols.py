"""Protocols.

These are the seams of the system. A new runtime / target is a new
class that satisfies these protocols. No inheritance, no abstract base
class — duck typing with static guarantees via `Protocol`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from ongoedge.core.types import (
    Detection,
    Frame,
    LatencySample,
    Precision,
    Target,
    TelemetrySnapshot,
)


@runtime_checkable
class ModelRuntime(Protocol):
    """Anything that can take a frame and produce detections + timing.

    Implementations: OnnxRuntime, TfliteRuntime, TensorRTRuntime, MockRuntime (tests).
    """

    @property
    def model_id(self) -> str: ...

    @property
    def target(self) -> Target: ...

    @property
    def precision(self) -> Precision: ...

    def warmup(self, n: int) -> None:
        """Run n synthetic forward passes. Called once before timing."""
        ...

    def infer(self, frame: Frame) -> tuple[list[Detection], LatencySample]:
        """Run inference. Returns detections and per-stage timing."""
        ...

    def close(self) -> None:
        """Release engine / device buffers. Idempotent."""
        ...


@runtime_checkable
class TelemetryProbe(Protocol):
    """Reads RAM / power / temperature from the device.

    On Jetson this wraps tegrastats; on Pi 5, vcgencmd. In tests we use
    a FixtureProbe that replays a deterministic trace.
    """

    @property
    def target(self) -> Target: ...

    def sample(self) -> TelemetrySnapshot: ...

    def stream(self, hz: float) -> Iterable[TelemetrySnapshot]:
        """Generator. Stops when the consumer stops iterating."""
        ...


@runtime_checkable
class Calibrator(Protocol):
    """For INT8 quantization: feeds representative data through the model."""

    def calibrate(self, model_path: Path, sample_frames: Iterable[Frame]) -> Path:
        """Returns path to the calibrated model artifact."""
        ...
