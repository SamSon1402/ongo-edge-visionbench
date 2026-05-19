"""FastAPI surface.

Mounts:
  GET  /healthz                — liveness probe
  GET  /api/v1/targets         — supported edge targets
  GET  /api/v1/models          — registered models
  POST /api/v1/runs            — kick off a sweep, returns run_id
  GET  /api/v1/runs/{run_id}   — fetch a run's result
  WS   /api/v1/runs/{run_id}/stream — live telemetry while running

The HTML dashboard (we ship it as a separate static artifact) calls these.
"""

from ongoedge.api.app import app, create_app

__all__ = ["app", "create_app"]
