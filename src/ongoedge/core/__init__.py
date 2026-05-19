"""Core domain types and protocols for ongoedge.

This module deliberately has no I/O dependencies. Everything here can be
imported without touching CUDA, ONNX, or hardware. Tests import from here.
"""

from ongoedge.core.errors import (
    BenchError,
    CalibrationError,
    EngineBuildError,
    TargetUnavailable,
)
from ongoedge.core.protocols import (
    Calibrator,
    ModelRuntime,
    TelemetryProbe,
)
from ongoedge.core.types import (
    BenchPlan,
    BenchResult,
    BoundingBox,
    Detection,
    Frame,
    LatencySample,
    Precision,
    Target,
    TelemetrySnapshot,
)

__all__ = [
    # types
    "BenchPlan",
    "BenchResult",
    "BoundingBox",
    "Detection",
    "Frame",
    "LatencySample",
    "Precision",
    "Target",
    "TelemetrySnapshot",
    # protocols
    "Calibrator",
    "ModelRuntime",
    "TelemetryProbe",
    # errors
    "BenchError",
    "CalibrationError",
    "EngineBuildError",
    "TargetUnavailable",
]
