"""Deterministic telemetry probe for tests + the host-cpu smoke benchmark.

Replays a configurable trace. Tests use this so they're hermetic — no
subprocess, no hardware, no flakes.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from ongoedge.core import Target, TelemetrySnapshot


class FixtureProbe:
    """Generates synthetic but plausible telemetry."""

    def __init__(
        self,
        *,
        target: Target = Target.HOST_CPU,
        ram_mb: float = 142.0,
        power_w: float = 4.5,
        temp_c: float = 64.0,
        noise: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._target = target
        self._ram = ram_mb
        self._power = power_w
        self._temp = temp_c
        self._noise = noise
        self._rng = random.Random(seed)
        self._tick = itertools.count()

    @property
    def target(self) -> Target:
        return self._target

    def sample(self) -> TelemetrySnapshot:
        n = next(self._tick)
        # Plausible quasi-periodic variation + gaussian noise.
        wobble = 0.5 + 0.5 * self._rng.gauss(0, 1)
        return TelemetrySnapshot(
            timestamp=datetime.now(UTC),
            ram_mb=self._ram + wobble * self._ram * self._noise,
            power_w=self._power + wobble * self._power * self._noise,
            soc_temp_c=self._temp + wobble * 2.0,
            gpu_util_pct=min(100.0, 65 + 10 * self._rng.random()),
            cpu_util_pct=min(100.0, 35 + 12 * self._rng.random()),
        )

    def stream(self, hz: float) -> Iterable[TelemetrySnapshot]:
        return _iter_fixture(self)


def _iter_fixture(probe: FixtureProbe) -> Iterator[TelemetrySnapshot]:
    while True:
        yield probe.sample()
