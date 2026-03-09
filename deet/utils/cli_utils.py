"""Utilities to make help integrate cli with rest of the codebase."""

from collections.abc import Generator, Iterable
from contextlib import contextmanager

import typer


@contextmanager
def optional_progress(
    iterable: Iterable, *, show_progress: bool = False, label: str = "Processing"
) -> Generator[Iterable, None, None]:
    """
    Context manager that yields an iterable.
    If show_progress is True, wraps it in a Typer progress bar.
    Otherwise, yields it unchanged.
    """
    if show_progress:
        with typer.progressbar(iterable, label=label) as progress:
            yield progress
    else:
        yield iterable
