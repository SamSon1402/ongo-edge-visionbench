"""API contract tests.

Exercise the actual endpoints with TestClient. No mocking — we hit
the real BackgroundTask path through to MockRuntime.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ongoedge.api import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_targets(client: TestClient) -> None:
    r = client.get("/api/v1/targets")
    assert r.status_code == 200
    assert "jetson-orin-nano" in r.json()


def test_list_models(client: TestClient) -> None:
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    ids = {m["model_id"] for m in r.json()}
    assert {"yolo11n", "mobilevit-xs"} <= ids


def test_create_run_rejects_unknown_model(client: TestClient) -> None:
    r = client.post(
        "/api/v1/runs",
        json={
            "target": "host-cpu",
            "model_ids": ["nonexistent-model"],
            "precisions": ["fp16"],
            "num_frames": 20,
        },
    )
    assert r.status_code == 422
    assert "nonexistent-model" in r.text


def test_create_and_fetch_run_end_to_end(client: TestClient) -> None:
    r = client.post(
        "/api/v1/runs",
        json={
            "target": "host-cpu",
            "model_ids": ["yolo11n"],
            "precisions": ["fp16"],
            "num_frames": 20,
            "warmup_frames": 2,
        },
    )
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    assert r.json()["plans_queued"] == 1

    # Poll until done. Mock runtime should finish well under 5s for 20 frames.
    deadline = time.time() + 8
    summary = None
    while time.time() < deadline:
        s = client.get(f"/api/v1/runs/{run_id}")
        assert s.status_code == 200
        summary = s.json()
        if summary["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)

    assert summary is not None
    assert summary["status"] == "done"
    assert summary["plans_done"] == 1
    assert len(summary["results"]) == 1
    res = summary["results"][0]
    assert res["model_id"] == "yolo11n"
    assert res["p50_ms"] > 0
    assert res["p99_ms"] >= res["p50_ms"]


def test_get_nonexistent_run_returns_404(client: TestClient) -> None:
    r = client.get("/api/v1/runs/does-not-exist")
    assert r.status_code == 404
