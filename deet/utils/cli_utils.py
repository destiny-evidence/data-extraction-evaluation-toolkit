"""Utilities to make help integrate cli with rest of the codebase."""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from enum import Enum
from typing import Any, NoReturn, get_args

import typer
from InquirerPy import inquirer
from loguru import logger
from pydantic import SecretStr
from pydantic.fields import FieldInfo

UNCHANGED_SECRET = "<unchanged>"  # noqa: S105


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


def inquire_pydantic_field(field: FieldInfo) -> str | int | float | None:
    """Prompt user to provide data for pydantic field."""
    widget_args: dict[str, Any] = {
        "message": field.description,
        "default": field.get_default(),
        "filter": lambda ans: ans.strip(),
    }
    extra = field.json_schema_extra
    if isinstance(extra, dict) and extra.get("skip_prompt"):
        return None
    if isinstance(field.annotation, type) and issubclass(field.annotation, Enum):
        widget_args["choices"] = [e.value for e in field.annotation]
        answer = inquirer.select(**widget_args).execute()
    elif field.annotation is float:
        widget_args["float_allowed"] = True
        answer = inquirer.number(**widget_args).execute()
    elif field.annotation is int or int in get_args(field.annotation):
        answer = inquirer.number(**widget_args).execute()
    elif field.annotation is SecretStr or SecretStr in get_args(field.annotation):
        if field.get_default() is None:
            widget_args["default"] = UNCHANGED_SECRET
        answer = inquirer.secret(**widget_args).execute()
        if answer == UNCHANGED_SECRET:
            answer = None
    else:
        answer = inquirer.text(**widget_args).execute()

    return answer
