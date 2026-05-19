"""Shared runtime helpers.

`Timed` is a context manager that measures wall-clock with `time.perf_counter`
in ms. We avoid `time.time()` because it can jump on NTP sync.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(slots=True)
class _Stopwatch:
    elapsed_ms: float = 0.0


@contextmanager
def timed() -> Iterator[_Stopwatch]:
    """Measure elapsed wall time in ms with perf_counter."""
    watch = _Stopwatch()
    start = time.perf_counter()
    try:
        yield watch
    finally:
        watch.elapsed_ms = (time.perf_counter() - start) * 1000.0


class RuntimeBase:
    """Tiny common base — DO NOT use for inheritance of behavior.

    It only provides `__enter__` / `__exit__` so callers can write
    `with OnnxRuntime(...) as rt: rt.infer(...)`. All real logic stays
    in the concrete runtime.
    """

    def __enter__(self) -> "RuntimeBase":
        return self

    def __exit__(self, *_: object) -> None:
        # Concrete runtimes override close(); we call it here.
        close = getattr(self, "close", None)
        if callable(close):
            close()
