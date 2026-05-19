"""Liveness / readiness."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Health(BaseModel):
    status: str
    version: str


@router.get("/healthz", response_model=Health)
def healthz() -> Health:
    return Health(status="ok", version="0.3.1")
