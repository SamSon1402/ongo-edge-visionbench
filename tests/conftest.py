"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from ongoedge.core import BenchPlan, Frame, Precision, Target


@pytest.fixture
def sample_frame() -> Frame:
    pixels = np.full((480, 640, 3), 128, dtype=np.uint8)
    return Frame(width=640, height=480, pixels=pixels)


@pytest.fixture
def basic_plan() -> BenchPlan:
    return BenchPlan(
        target=Target.HOST_CPU,
        model_id="yolo11n",
        precision=Precision.FP16,
        input_h=480,
        input_w=640,
        num_frames=20,
        warmup_frames=2,
    )
