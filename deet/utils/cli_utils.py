"""Utilities to make help integrate cli with rest of the codebase."""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any, NoReturn

import typer
from loguru import logger


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


def echo_and_log(message: Any, **kwargs) -> None:  # noqa: ANN401
    """
    Echo (in typer) and log (via logger) a message simultaenously.

    NOTE: pass typer-style stuff via kwargs.
    """
    typer.secho(message, **kwargs)
    logger.info(f"typer .secho: {message}")


def fail_with_message(message: str) -> NoReturn:
    """Print message and exit CLI."""
    echo_and_log(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
