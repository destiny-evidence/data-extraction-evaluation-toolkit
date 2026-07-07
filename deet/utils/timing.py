"""Wall-clock timing helpers."""

import time
from dataclasses import dataclass
from typing import Self


@dataclass
class PerfTimer:
    """Context manager that records elapsed wall time in seconds."""

    seconds: float = 0.0

    def __enter__(self) -> Self:
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_args: object) -> None:
        """Stop the timer and store elapsed seconds (3 decimal places)."""
        self.seconds = round(time.perf_counter() - self._start, 3)
