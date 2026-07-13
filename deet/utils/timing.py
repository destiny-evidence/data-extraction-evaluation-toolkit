"""Wall-clock timing helpers."""

import time
from collections.abc import Generator
from contextlib import contextmanager

from pydantic import BaseModel, Field


class ElapsedSeconds(BaseModel):
    """Elapsed wall-clock time in seconds, populated when a timing context exits."""

    seconds: float = Field(default=0.0, ge=0.0)


@contextmanager
def measure_elapsed() -> Generator[ElapsedSeconds, None, None]:
    """Yield an ``ElapsedSeconds`` model with ``seconds`` set on context exit."""
    elapsed = ElapsedSeconds()
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.seconds = round(time.perf_counter() - start, 3)
