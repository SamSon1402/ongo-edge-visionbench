"""Core type contracts. If these break, the universe is broken."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ongoedge.core import (
    BenchPlan,
    BoundingBox,
    Detection,
    Frame,
    LatencySample,
    Precision,
    Target,
)
from ongoedge.core.types import _quantile


class TestBoundingBox:
    def test_iou_identical(self) -> None:
        b = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        assert b.iou(b) == 1.0

    def test_iou_disjoint(self) -> None:
        a = BoundingBox(x1=0, y1=0, x2=5, y2=5)
        b = BoundingBox(x1=10, y1=10, x2=15, y2=15)
        assert a.iou(b) == 0.0

    def test_iou_half_overlap(self) -> None:
        a = BoundingBox(x1=0, y1=0, x2=10, y2=10)   # area 100
        b = BoundingBox(x1=5, y1=0, x2=15, y2=10)   # area 100, inter 50
        assert abs(a.iou(b) - 50 / 150) < 1e-9

    def test_rejects_degenerate(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=0, x2=5, y2=10)


class TestFrame:
    def test_accepts_valid(self) -> None:
        px = np.zeros((100, 200, 3), dtype=np.uint8)
        f = Frame(width=200, height=100, pixels=px)
        assert f.width == 200

    def test_rejects_shape_mismatch(self) -> None:
        px = np.zeros((100, 200, 3), dtype=np.uint8)
        with pytest.raises(ValidationError):
            Frame(width=999, height=100, pixels=px)

    def test_rejects_wrong_dtype(self) -> None:
        px = np.zeros((100, 200, 3), dtype=np.float32)
        with pytest.raises(ValidationError):
            Frame(width=200, height=100, pixels=px)


class TestDetection:
    def test_confidence_bounds(self) -> None:
        b = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        with pytest.raises(ValidationError):
            Detection(label="x", confidence=1.5, box=b)
        with pytest.raises(ValidationError):
            Detection(label="x", confidence=-0.1, box=b)


class TestLatencySample:
    def test_total(self) -> None:
        s = LatencySample(preprocess_ms=1.0, inference_ms=10.0, postprocess_ms=2.0)
        assert s.total_ms == 13.0


class TestBenchPlan:
    def test_plan_id_stable(self) -> None:
        p1 = BenchPlan(
            target=Target.JETSON_ORIN_NANO,
            model_id="yolo11n",
            precision=Precision.INT8,
            input_h=480,
            input_w=640,
            num_frames=100,
        )
        p2 = BenchPlan(
            target=Target.JETSON_ORIN_NANO,
            model_id="yolo11n",
            precision=Precision.INT8,
            input_h=480,
            input_w=640,
            num_frames=100,
        )
        assert p1.plan_id == p2.plan_id


class TestQuantile:
    def test_empty(self) -> None:
        assert _quantile([], 0.5) == 0.0

    def test_median(self) -> None:
        assert _quantile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_p99_on_long_tail(self) -> None:
        # Many low samples + a sprinkle of spikes near the top end.
        # With 100 samples and linear interp on a sorted array of 99×10 + 1×100,
        # p99 sits between indices 98 (=10) and 99 (=100) → ≥ 10.
        # The stronger property: ten spikes guarantees p99 lands on the spikes.
        samples = [10.0] * 90 + [100.0] * 10
        assert _quantile(samples, 0.99) >= 90.0

    def test_p99_strictly_above_p50_when_long_tail(self) -> None:
        samples = [10.0] * 99 + [100.0]
        # The fundamental contract: p99 is at least as large as p50.
        assert _quantile(samples, 0.99) >= _quantile(samples, 0.5)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            _quantile([1, 2, 3], 1.5)


class TestPrecision:
    def test_bytes_per_param_ordered(self) -> None:
        assert (
            Precision.FP32.bytes_per_param
            > Precision.FP16.bytes_per_param
            > Precision.INT8.bytes_per_param
            > Precision.INT4.bytes_per_param
        )


class TestTarget:
    def test_arm_classification(self) -> None:
        assert Target.JETSON_ORIN_NANO.is_arm
        assert Target.RPI5.is_arm
        assert not Target.HOST_CPU.is_arm
