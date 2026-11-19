"""Interface to pass messages to UI(s)."""

from typing import NoReturn

import typer

from deet.logger import logger
from deet.settings import LogLevel
from deet.ui.terminal import render_to_console


def notify(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Send messages to logger and to UIs (currently the console)."""
    logger.bind(is_echo=True).log(level.value, message)
    render_to_console(message, level)


def fail_with_message(message: str) -> NoReturn:
    """
    Print message and exit CLI.

    Deprecated: raise a ``RunAbortingError`` instead. ``deet.scripts.cli.main``
    prints any ``UserFacingError`` and exits with code 1, and the error also
    works when the code is called outside the CLI.
    """
    notify(message, level=LogLevel.ERROR)
    raise typer.Exit(code=1)
