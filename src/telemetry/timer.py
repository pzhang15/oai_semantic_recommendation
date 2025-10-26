from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Tuple


class Span:
    def __init__(self, name: str) -> None:
        self.name = name
        self.start: float | None = None
        self.end: float | None = None
        self.ms: float = 0.0

    def __enter__(self) -> "Span":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()
        self.ms = (self.end - (self.start or self.end)) * 1000.0


def span(name: str) -> Span:
    return Span(name)


def time_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    end = time.perf_counter()
    return res, (end - start) * 1000.0


