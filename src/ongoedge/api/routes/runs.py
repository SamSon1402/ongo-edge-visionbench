"""Run endpoints.

POST /runs           kick off a sweep (returns run_id)
GET  /runs/{id}      fetch the result
GET  /runs/{id}/ws   live telemetry while running

The run store is process-local for the skeleton. Production: swap for
Redis or postgres behind the same `RunStore` interface.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ongoedge import models as model_registry
from ongoedge.bench import Runner
from ongoedge.core import (
    BenchPlan,
    BenchResult,
    Frame,
    Precision,
    Target,
)
from ongoedge.runtimes.mock import MockRuntime
from ongoedge.telemetry.fixture import FixtureProbe

router = APIRouter()
log = structlog.get_logger(__name__)


# ── request / response schemas ───────────────────────────────────────


class CreateRunRequest(BaseModel):
    target: Target
    model_ids: list[str] = Field(min_length=1, max_length=20)
    precisions: list[Precision] = Field(min_length=1, max_length=4)
    num_frames: int = Field(default=200, ge=10, le=5000)
    warmup_frames: int = Field(default=10, ge=0, le=200)
    input_h: int = Field(default=480, gt=0)
    input_w: int = Field(default=640, gt=0)


class CreateRunResponse(BaseModel):
    run_id: str
    plans_queued: int
    queued_at: datetime


class RunSummary(BaseModel):
    run_id: str
    status: str  # queued | running | done | failed
    target: Target
    plans_total: int
    plans_done: int
    results: list[dict]  # serialized BenchResult subset


# ── in-memory store (replace with redis/pg for prod) ────────────────


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def create(self, summary: dict) -> None:
        async with self._lock:
            self._runs[summary["run_id"]] = summary

    async def update(self, run_id: str, **patch) -> None:
        async with self._lock:
            if run_id not in self._runs:
                raise KeyError(run_id)
            self._runs[run_id].update(patch)

    async def get(self, run_id: str) -> dict | None:
        async with self._lock:
            return self._runs.get(run_id)

    async def append_result(self, run_id: str, result: BenchResult) -> None:
        async with self._lock:
            entry = self._runs[run_id]
            entry["results"].append(_serialize_result(result))
            entry["plans_done"] = len(entry["results"])


_STORE = RunStore()


def get_store() -> RunStore:
    return _STORE


# ── endpoints ───────────────────────────────────────────────────────


@router.get("/targets")
def list_targets() -> list[str]:
    return [t.value for t in Target]


@router.get("/models")
def list_models() -> list[dict]:
    return [
        {
            "model_id": m.model_id,
            "description": m.description,
            "input_h": m.input_h,
            "input_w": m.input_w,
        }
        for m in model_registry.REGISTRY.values()
    ]


@router.post("/runs", response_model=CreateRunResponse, status_code=202)
async def create_run(
    body: CreateRunRequest,
    background: BackgroundTasks,
    store: Annotated[RunStore, Depends(get_store)],
) -> CreateRunResponse:
    # Validate model ids up front — better a 422 here than a silent skip later.
    unknown = [m for m in body.model_ids if m not in model_registry.REGISTRY]
    if unknown:
        raise HTTPException(422, detail=f"unknown model_ids: {unknown}")

    run_id = uuid.uuid4().hex[:12]
    plans_total = len(body.model_ids) * len(body.precisions)
    await store.create(
        {
            "run_id": run_id,
            "status": "queued",
            "target": body.target.value,
            "plans_total": plans_total,
            "plans_done": 0,
            "results": [],
            "queued_at": datetime.now(UTC).isoformat(),
        }
    )

    background.add_task(_execute_run, run_id, body, store)
    log.info("api.run_queued", run_id=run_id, plans=plans_total, target=body.target.value)
    return CreateRunResponse(
        run_id=run_id,
        plans_queued=plans_total,
        queued_at=datetime.now(UTC),
    )


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: str,
    store: Annotated[RunStore, Depends(get_store)],
) -> RunSummary:
    summary = await store.get(run_id)
    if summary is None:
        raise HTTPException(404, detail=f"run {run_id} not found")
    return RunSummary(**summary)


@router.websocket("/runs/{run_id}/stream")
async def stream_run(
    websocket: WebSocket,
    run_id: str,
) -> None:
    """Push run progress to the dashboard every 250ms while the run is live."""
    await websocket.accept()
    try:
        while True:
            summary = await _STORE.get(run_id)
            if summary is None:
                await websocket.send_json({"error": "not_found"})
                break
            await websocket.send_json(
                {
                    "run_id": run_id,
                    "status": summary["status"],
                    "plans_done": summary["plans_done"],
                    "plans_total": summary["plans_total"],
                    "latest": summary["results"][-1] if summary["results"] else None,
                }
            )
            if summary["status"] in {"done", "failed"}:
                break
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        log.info("api.ws_disconnect", run_id=run_id)


# ── background executor ─────────────────────────────────────────────


async def _execute_run(run_id: str, req: CreateRunRequest, store: RunStore) -> None:
    """The actual benchmark loop. Runs in a FastAPI BackgroundTask.

    We use MockRuntime + FixtureProbe here because the API server itself
    won't be the Jetson — the runner ships as a separate worker process
    bound to the target. The mock path lets the dashboard be developed
    against a real-shaped API.
    """
    await store.update(run_id, status="running", started_at=datetime.now(UTC).isoformat())
    try:
        for model_id in req.model_ids:
            for precision in req.precisions:
                plan = BenchPlan(
                    target=req.target,
                    model_id=model_id,
                    precision=precision,
                    input_h=req.input_h,
                    input_w=req.input_w,
                    num_frames=req.num_frames,
                    warmup_frames=req.warmup_frames,
                )
                runner = Runner(
                    runtime=MockRuntime(
                        model_id=model_id,
                        target=req.target,
                        precision=precision,
                        baseline_ms=_baseline_for(model_id, precision),
                        seed=hash((run_id, model_id, precision)) & 0xFFFF,
                    ),
                    probe=FixtureProbe(target=req.target),
                    frame_source=_synthetic_frame,
                )
                # Run in a worker thread; the bench loop is CPU-bound.
                result = await asyncio.to_thread(runner.execute, plan)
                await store.append_result(run_id, result)

        await store.update(run_id, status="done", finished_at=datetime.now(UTC).isoformat())
    except Exception as exc:  # noqa: BLE001
        log.exception("api.run_failed", run_id=run_id, error=str(exc))
        await store.update(run_id, status="failed", error=str(exc))


# ── helpers ─────────────────────────────────────────────────────────


def _baseline_for(model_id: str, precision: Precision) -> float:
    """Plausible baseline latency for the mock runtime."""
    base = {
        "yolo11n": 11.5,
        "yolov8n": 12.7,
        "nanodet-plus": 9.0,
        "mobilevit-xs": 18.0,
        "efficientvit-b0": 15.5,
    }.get(model_id, 14.0)
    multiplier = {Precision.FP32: 1.6, Precision.FP16: 1.0, Precision.INT8: 0.7, Precision.INT4: 0.55}
    return base * multiplier[precision]


def _synthetic_frame(plan: BenchPlan) -> Frame:
    """Deterministic dummy frame keyed off plan dims."""
    pixels = np.full((plan.input_h, plan.input_w, 3), 96, dtype=np.uint8)
    return Frame(width=plan.input_w, height=plan.input_h, pixels=pixels)


def _serialize_result(r: BenchResult) -> dict:
    return {
        "plan_id": r.plan.plan_id,
        "model_id": r.plan.model_id,
        "precision": r.plan.precision.value,
        "p50_ms": round(r.p50_ms, 2),
        "p99_ms": round(r.p99_ms, 2),
        "fps": round(r.fps, 1),
        "peak_ram_mb": round(r.peak_ram_mb, 1),
        "mean_power_w": round(r.mean_power_w, 2),
        "frontier_score": round(r.frontier_score(), 4),
    }
