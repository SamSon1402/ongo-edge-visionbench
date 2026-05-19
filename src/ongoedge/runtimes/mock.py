"""Mock runtime — for tests and the host-cpu smoke benchmark.

Behaves like a real model: warmup costs a bit, inference takes a configurable
amount of time with realistic noise (long-tailed: most frames fast, occasional
spikes), produces plausible-looking detections.

Critically: deterministic given a seed, so tests are stable.
"""

from __future__ import annotations

import random
import time

from ongoedge.core import (
    BoundingBox,
    Detection,
    Frame,
    LatencySample,
    Precision,
    Target,
)
from ongoedge.runtimes.base import RuntimeBase, timed


class MockRuntime(RuntimeBase):
    """Simulates a small object-detector with realistic timing distribution."""

    def __init__(
        self,
        *,
        model_id: str = "mock-yolo",
        target: Target = Target.HOST_CPU,
        precision: Precision = Precision.FP16,
        baseline_ms: float = 12.0,
        spike_probability: float = 0.04,
        seed: int = 0,
    ) -> None:
        self._model_id = model_id
        self._target = target
        self._precision = precision
        self._baseline_ms = baseline_ms
        self._spike_probability = spike_probability
        self._rng = random.Random(seed)
        self._closed = False

    # ── ModelRuntime protocol ──────────────────────────────────────

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
        # Warmup is real but cheap. We don't sleep — we just spin briefly
        # so unit tests stay fast.
        for _ in range(n):
            _ = self._rng.random()

    def infer(self, frame: Frame) -> tuple[list[Detection], LatencySample]:
        if self._closed:
            raise RuntimeError("MockRuntime: infer() called after close()")

        # Realistic timing: gaussian around baseline + rare long-tail spike.
        with timed() as t_pre:
            self._busy(0.5)

        with timed() as t_inf:
            jitter = self._rng.gauss(0, self._baseline_ms * 0.08)
            spike = (
                self._rng.uniform(self._baseline_ms, self._baseline_ms * 3.0)
                if self._rng.random() < self._spike_probability
                else 0.0
            )
            self._busy(max(0.5, self._baseline_ms + jitter + spike))

        with timed() as t_post:
            detections = self._fake_detections(frame)

        return detections, LatencySample(
            preprocess_ms=t_pre.elapsed_ms,
            inference_ms=t_inf.elapsed_ms,
            postprocess_ms=t_post.elapsed_ms,
        )

    def close(self) -> None:
        self._closed = True

    # ── helpers ────────────────────────────────────────────────────

    def _busy(self, target_ms: float) -> None:
        """Busy-wait for approximately target_ms. Sleep is too coarse here."""
        deadline = time.perf_counter() + target_ms / 1000.0
        # A small spin to consume time without yielding (closer to a real
        # GPU kernel call from the host's perspective).
        while time.perf_counter() < deadline:
            pass

    def _fake_detections(self, frame: Frame) -> list[Detection]:
        n = self._rng.choices([0, 1, 2, 3], weights=[2, 5, 4, 1])[0]
        out: list[Detection] = []
        for _ in range(n):
            x1 = self._rng.uniform(0, frame.width - 32)
            y1 = self._rng.uniform(0, frame.height - 32)
            w = self._rng.uniform(16, min(frame.width - x1, 96))
            h = self._rng.uniform(16, min(frame.height - y1, 96))
            out.append(
                Detection(
                    label=self._rng.choice(["person", "hand", "object", "face"]),
                    confidence=self._rng.uniform(0.55, 0.97),
                    box=BoundingBox(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h),
                )
            )
        return out
