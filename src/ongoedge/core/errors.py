"""Error taxonomy.

Bare exceptions are a smell. Anything we raise has a class so callers
can pattern-match instead of parsing strings.
"""

from __future__ import annotations


class BenchError(Exception):
    """Base class for everything raised from inside ongoedge."""


class TargetUnavailable(BenchError):
    """Requested edge target isn't on this host (e.g. no Jetson detected)."""


class EngineBuildError(BenchError):
    """A runtime failed to build / load the model (bad ONNX, missing TRT)."""


class CalibrationError(BenchError):
    """INT8 calibration failed (not enough samples, NaN activations, etc.)."""


class TelemetryUnavailable(BenchError):
    """Device telemetry can't be read (missing tegrastats, permissions, …)."""
