"""FastAPI application factory.

`create_app()` returns a fresh app — used by tests so they don't share state.
`app` is the module-level instance uvicorn loads.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ongoedge.api.routes import health, runs


def create_app() -> FastAPI:
    app = FastAPI(
        title="OngoEdge-VisionBench API",
        version="0.3.1",
        description=(
            "Edge CV benchmark backend for the Ongo companion robot. "
            "Drives sweeps across Jetson / RPi5 / Coral / RK3588 targets."
        ),
    )

    # Dashboard is served from a different origin in dev. In prod we sit
    # behind the same reverse proxy, but CORS is cheap insurance.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:8000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["health"])
    app.include_router(runs.router, prefix="/api/v1", tags=["runs"])
    return app


app = create_app()
