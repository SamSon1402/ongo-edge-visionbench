"""Jetson telemetry — wraps `tegrastats` subprocess output.

A tegrastats line looks like (JetPack 6):

    RAM 1839/7763MB SWAP 0/3881MB ... CPU [22%@1497,15%@1497,...] GR3D_FREQ 0%@[420]
    APE 174 MTS fg 0% bg 0% AO@45.5C GPU@46.0C CPU@46.5C SOC2@45.5C SOC1@45.0C
    SOC0@46.0C ... VDD_IN 4521mW/4521mW VDD_CPU_GPU_CV 1320mW/1320mW
    VDD_SOC 1080mW/1080mW

We pull RAM, VDD_IN (= total power), and the max temperature across SoC zones.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Final

import structlog

from ongoedge.core import Target, TelemetrySnapshot
from ongoedge.core.errors import TelemetryUnavailable

log = structlog.get_logger(__name__)


# Regexes compiled once — these are hot.
_RAM_RE:    Final = re.compile(r"RAM\s+(?P<used>\d+)/(?P<total>\d+)MB")
_VDDIN_RE:  Final = re.compile(r"VDD_IN\s+(?P<inst>\d+)mW")
_TEMP_RE:   Final = re.compile(r"(?P<zone>[A-Z0-9]+)@(?P<deg>\d+(?:\.\d+)?)C")
_GPUUTIL_RE: Final = re.compile(r"GR3D_FREQ\s+(?P<pct>\d+)%")
_CPUBLK_RE: Final = re.compile(r"CPU \[(?P<inner>[^\]]+)\]")


class JetsonProbe:
    """Reads tegrastats. Two modes: single-shot and streaming."""

    def __init__(self, *, target: Target = Target.JETSON_ORIN_NANO, interval_ms: int = 250) -> None:
        if target not in {Target.JETSON_ORIN_NANO, Target.JETSON_ORIN_NX}:
            raise TelemetryUnavailable(f"JetsonProbe doesn't support {target}")
        self._target = target
        self._interval_ms = interval_ms

    @property
    def target(self) -> Target:
        return self._target

    def sample(self) -> TelemetrySnapshot:
        """One-shot read: launches tegrastats, grabs the first line, kills it."""
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except FileNotFoundError as exc:
            raise TelemetryUnavailable(
                "tegrastats not found on PATH. Is this a Jetson with JetPack installed?"
            ) from exc

        try:
            assert proc.stdout is not None
            line = proc.stdout.readline()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        return self._parse(line)

    def stream(self, hz: float = 4.0) -> Iterable[TelemetrySnapshot]:
        """Long-lived stream. Caller breaks the loop to stop tegrastats."""
        interval_ms = int(1000.0 / max(hz, 0.1))
        return _iter_tegrastats(interval_ms, self._parse)

    # ── parser ─────────────────────────────────────────────────────

    @staticmethod
    def _parse(line: str) -> TelemetrySnapshot:
        if not line:
            raise TelemetryUnavailable("empty tegrastats line")

        ram_used = _find_int(_RAM_RE, line, "used")
        power_mw = _find_int(_VDDIN_RE, line, "inst", default=0)
        temps = [float(m.group("deg")) for m in _TEMP_RE.finditer(line)]
        max_temp = max(temps) if temps else 0.0
        gpu_util = _find_int(_GPUUTIL_RE, line, "pct", default=None)

        # CPU block looks like "22%@1497,15%@1497,...". Parse percentages.
        cpu_util: float | None = None
        cpu_block = _CPUBLK_RE.search(line)
        if cpu_block:
            try:
                vals = [
                    float(p.split("%")[0])
                    for p in cpu_block.group("inner").split(",")
                    if "%" in p
                ]
                cpu_util = sum(vals) / len(vals) if vals else None
            except ValueError:
                cpu_util = None

        return TelemetrySnapshot(
            timestamp=datetime.now(UTC),
            ram_mb=float(ram_used),
            power_w=power_mw / 1000.0,
            soc_temp_c=max_temp,
            gpu_util_pct=float(gpu_util) if gpu_util is not None else None,
            cpu_util_pct=cpu_util,
        )


def _find_int(pattern: re.Pattern[str], text: str, group: str, default: int | None = None) -> int:
    m = pattern.search(text)
    if not m:
        if default is None:
            raise TelemetryUnavailable(f"tegrastats line missing {pattern.pattern!r}")
        return default
    return int(m.group(group))


def _iter_tegrastats(
    interval_ms: int, parser: callable
) -> Iterator[TelemetrySnapshot]:
    proc = subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                yield parser(line)
            except TelemetryUnavailable:
                log.warning("telemetry.bad_line", line=line[:120])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
