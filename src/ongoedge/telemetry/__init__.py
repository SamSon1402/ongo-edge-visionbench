"""Device telemetry probes — concrete implementations of TelemetryProbe."""

from ongoedge.telemetry.fixture import FixtureProbe
from ongoedge.telemetry.jetson import JetsonProbe

__all__ = ["FixtureProbe", "JetsonProbe"]
