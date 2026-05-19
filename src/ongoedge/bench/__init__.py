"""Benchmark engine: takes a BenchPlan, returns a BenchResult."""

from ongoedge.bench.runner import Runner
from ongoedge.bench.sweep import Sweep

__all__ = ["Runner", "Sweep"]
