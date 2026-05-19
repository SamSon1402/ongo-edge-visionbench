"""Sweep: cartesian product of (models × precisions) executed in series.

A sweep is just a list of BenchPlans + a place to put the results. We don't
parallelise across plans because we'd be measuring the wrong thing — the
runs must not contend for the device under test.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import structlog

from ongoedge.bench.runner import Runner
from ongoedge.core import BenchPlan, BenchResult, Precision, Target

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class Sweep:
    """A planned set of benchmark runs against one target."""

    target: Target
    model_ids: list[str]
    precisions: list[Precision]
    input_h: int = 480
    input_w: int = 640
    num_frames: int = 500
    warmup_frames: int = 10
    results: list[BenchResult] = field(default_factory=list)

    @property
    def plans(self) -> Iterable[BenchPlan]:
        for model_id in self.model_ids:
            for precision in self.precisions:
                yield BenchPlan(
                    target=self.target,
                    model_id=model_id,
                    precision=precision,
                    input_h=self.input_h,
                    input_w=self.input_w,
                    num_frames=self.num_frames,
                    warmup_frames=self.warmup_frames,
                )

    def execute(self, build_runner: "RunnerFactory") -> list[BenchResult]:
        """Run every plan, collect every result. Failures are logged, not raised."""
        for plan in self.plans:
            try:
                runner = build_runner(plan)
                self.results.append(runner.execute(plan))
            except Exception as exc:  # noqa: BLE001
                log.exception("sweep.plan_failed", plan_id=plan.plan_id, error=str(exc))
        return self.results

    def pareto_frontier(self) -> list[BenchResult]:
        """Return results sorted by frontier_score, best first."""
        return sorted(self.results, key=lambda r: r.frontier_score(), reverse=True)


# A factory builds a Runner from a Plan. Defined as a Protocol-shaped callable.
from typing import Protocol  # noqa: E402


class RunnerFactory(Protocol):
    def __call__(self, plan: BenchPlan) -> Runner: ...
