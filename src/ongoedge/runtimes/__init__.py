"""Runtime adapters. Each implements the ModelRuntime protocol."""

from ongoedge.runtimes.base import RuntimeBase
from ongoedge.runtimes.mock import MockRuntime
from ongoedge.runtimes.onnx import OnnxRuntime

__all__ = ["RuntimeBase", "MockRuntime", "OnnxRuntime"]
