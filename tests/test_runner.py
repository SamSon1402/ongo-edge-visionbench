"""Runner contract tests.

These pin down the most important behaviors of the benchmark loop:
  * Warmup is excluded from results.
  * The result has exactly num_frames latency samples on success.
  * Telemetry actually gets collected.
  * Target mismatch between runtime and probe is rejected up front.
  * If the runtime raises mid-run, we get a partial result, not a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from ongoedge.bench import Runner
from ongoedge.core import BenchError, BenchPlan, Frame, Precision, Target
from ongoedge.runtimes.mock import MockRuntime
from ongoedge.telemetry.fixture import FixtureProbe


def _frame_for(plan: BenchPlan) -> Frame:
    px = np.full((plan.input_h, plan.input_w, 3), 128, dtype=np.uint8)
    return Frame(width=plan.input_w, height=plan.input_h, pixels=px)


def test_runner_produces_exactly_num_frames_samples(basic_plan: BenchPlan) -> None:
    runner = Runner(
        runtime=MockRuntime(baseline_ms=1.0),
        probe=FixtureProbe(),
        frame_source=_frame_for,
    )
    result = runner.execute(basic_plan)
    assert len(result.latencies) == basic_plan.num_frames


def test_runner_rejects_mismatched_targets(basic_plan: BenchPlan) -> None:
    with pytest.raises(BenchError):
        Runner(
            runtime=MockRuntime(target=Target.HOST_CPU),
            probe=FixtureProbe(target=Target.JETSON_ORIN_NANO),
            frame_source=_frame_for,
        )


def test_runner_collects_some_telemetry(basic_plan: BenchPlan) -> None:
    # Use enough frames so the 4Hz sampler can fire at least once.
    plan = basic_plan.model_copy(update={"num_frames": 200})
    runner = Runner(
        runtime=MockRuntime(baseline_ms=1.0),
        probe=FixtureProbe(),
        frame_source=_frame_for,
        telemetry_hz=10.0,
    )
    result = runner.execute(plan)
    assert len(result.telemetry) >= 1
    assert result.peak_ram_mb > 0
    assert result.mean_power_w > 0


def test_runner_p50_p99_consistent(basic_plan: BenchPlan) -> None:
    plan = basic_plan.model_copy(update={"num_frames": 100})
    runner = Runner(
        runtime=MockRuntime(baseline_ms=8.0, spike_probability=0.0, seed=42),
        probe=FixtureProbe(),
        frame_source=_frame_for,
    )
    result = runner.execute(plan)
    assert result.p50_ms <= result.p99_ms
    assert result.fps > 0


def test_runner_partial_result_when_runtime_raises(basic_plan: BenchPlan) -> None:
    """A runtime that explodes mid-way should still yield a partial result."""

    class FlakyRuntime(MockRuntime):
        _count = 0

        def infer(self, frame):
            type(self)._count += 1
            if type(self)._count == 5:
                raise RuntimeError("synthetic boom")
            return super().infer(frame)

    runner = Runner(
        runtime=FlakyRuntime(baseline_ms=1.0),
        probe=FixtureProbe(),
        frame_source=_frame_for,
    )
    result = runner.execute(basic_plan.model_copy(update={"num_frames": 20}))
    # We got *some* samples before the crash, but not all of them.
    assert 0 < len(result.latencies) < 20
