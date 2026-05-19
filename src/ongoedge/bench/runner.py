"""Benchmark runner.

Executes a single BenchPlan against one (runtime, probe) pair and produces a
BenchResult. No I/O beyond logging — the runtime and probe encapsulate that.

Design rules followed here:
  * Plan in, Result out. No mutation of inputs.
  * Telemetry runs in a thread, sampled at a fixed cadence, joined cleanly.
  * Warmup is excluded from the result *by construction* — it never enters
    the latencies list. This is a frequent source of fudged numbers.
  * If anything raises mid-run we still return the partial result so the
    operator can see where it died.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime

import structlog

from ongoedge.core import (
    BenchError,
    BenchPlan,
    BenchResult,
    Frame,
    LatencySample,
    ModelRuntime,
    TelemetryProbe,
    TelemetrySnapshot,
)

log = structlog.get_logger(__name__)


class Runner:
    """Executes one BenchPlan."""

    def __init__(
        self,
        runtime: ModelRuntime,
        probe: TelemetryProbe,
        frame_source: Callable[[BenchPlan], Frame],
        telemetry_hz: float = 4.0,
    ) -> None:
        if runtime.target != probe.target:
            raise BenchError(
                f"runtime target {runtime.target} ≠ probe target {probe.target}"
            )
        self._runtime = runtime
        self._probe = probe
        self._frame_source = frame_source
        self._telemetry_hz = telemetry_hz

    # ── public API ──────────────────────────────────────────────────

    def execute(self, plan: BenchPlan) -> BenchResult:
        log.info("bench.start", plan_id=plan.plan_id, frames=plan.num_frames)
        started_at = datetime.now(UTC)

        # Warmup. These iterations do NOT enter the result.
        if plan.warmup_frames > 0:
            log.debug("bench.warmup", n=plan.warmup_frames)
            self._runtime.warmup(plan.warmup_frames)

        # Telemetry sampler runs in the background while we infer.
        snapshots: list[TelemetrySnapshot] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=self._sample_loop, args=(snapshots, stop), daemon=True
        )
        sampler.start()

        latencies: list[LatencySample] = []
        try:
            for i in range(plan.num_frames):
                frame = self._frame_source(plan)
                _, sample = self._runtime.infer(frame)
                latencies.append(sample)
                if (i + 1) % 50 == 0:
                    log.debug("bench.progress", done=i + 1, total=plan.num_frames)
        except Exception as exc:  # noqa: BLE001 - we *want* to capture & return
            log.exception("bench.runtime_failed", error=str(exc))
            # Fall through: we still return what we have.
        finally:
            stop.set()
            sampler.join(timeout=2.0)
            try:
                self._runtime.close()
            except Exception:  # noqa: BLE001
                log.warning("bench.close_failed")

        finished_at = datetime.now(UTC)
        result = BenchResult(
            plan=plan,
            started_at=started_at,
            finished_at=finished_at,
            latencies=latencies,
            telemetry=snapshots,
            mean_ap_50=None,  # filled by an eval pass if a labeled set is provided
        )
        log.info(
            "bench.done",
            plan_id=plan.plan_id,
            p50=round(result.p50_ms, 2),
            p99=round(result.p99_ms, 2),
            fps=round(result.fps, 1),
            ram_mb=round(result.peak_ram_mb, 1),
            power_w=round(result.mean_power_w, 2),
        )
        return result

    # ── internal ────────────────────────────────────────────────────

    def _sample_loop(
        self, sink: list[TelemetrySnapshot], stop: threading.Event
    ) -> None:
        period = 1.0 / max(self._telemetry_hz, 0.1)
        while not stop.is_set():
            try:
                sink.append(self._probe.sample())
            except Exception:  # noqa: BLE001
                log.warning("telemetry.sample_failed")
            stop.wait(period)
