"""Jetson tegrastats parser.

We pin the regex against a real captured tegrastats line so it never
silently regresses when JetPack changes the output format.
"""

from __future__ import annotations

import pytest

from ongoedge.core.errors import TelemetryUnavailable
from ongoedge.telemetry.jetson import JetsonProbe

# Real-ish tegrastats line from JetPack 6 / Orin Nano.
SAMPLE_LINE = (
    "12-31-2024 11:22:33 RAM 1839/7763MB (lfb 4x2MB) SWAP 0/3881MB (cached 0MB) "
    "CPU [22%@1497,15%@1497,18%@1497,21%@1497,17%@1497,19%@1497] EMC_FREQ 0%@2133 "
    "GR3D_FREQ 14%@[420] APE 174 MTS fg 0% bg 0% AO@45.5C GPU@46.0C CPU@46.5C "
    "SOC2@45.5C SOC1@45.0C SOC0@46.0C TJ@46.5C VDD_IN 4521mW/4521mW "
    "VDD_CPU_GPU_CV 1320mW/1320mW VDD_SOC 1080mW/1080mW"
)


def test_parser_extracts_ram() -> None:
    snap = JetsonProbe._parse(SAMPLE_LINE)
    assert snap.ram_mb == 1839.0


def test_parser_extracts_power_in_watts() -> None:
    snap = JetsonProbe._parse(SAMPLE_LINE)
    # VDD_IN is 4521mW → 4.521W
    assert abs(snap.power_w - 4.521) < 1e-6


def test_parser_extracts_max_temperature() -> None:
    snap = JetsonProbe._parse(SAMPLE_LINE)
    # Max of AO/GPU/CPU/SOC zones is 46.5°C
    assert snap.soc_temp_c == 46.5


def test_parser_extracts_gpu_util() -> None:
    snap = JetsonProbe._parse(SAMPLE_LINE)
    assert snap.gpu_util_pct == 14.0


def test_parser_extracts_cpu_util_as_mean() -> None:
    snap = JetsonProbe._parse(SAMPLE_LINE)
    # Mean of [22, 15, 18, 21, 17, 19] = 18.666...
    assert snap.cpu_util_pct is not None
    assert abs(snap.cpu_util_pct - 18.666666666666668) < 1e-6


def test_parser_rejects_empty() -> None:
    with pytest.raises(TelemetryUnavailable):
        JetsonProbe._parse("")


def test_parser_rejects_unrelated_line() -> None:
    with pytest.raises(TelemetryUnavailable):
        JetsonProbe._parse("nothing useful here\n")
