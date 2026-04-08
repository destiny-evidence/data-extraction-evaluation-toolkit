"""decorators to handle typer context in commands."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import ParamSpec, TypeVar, cast

import typer

from deet.data_models.project import DeetProject
from deet.ui import fail_with_message


@dataclass
class CLIState:
    """Structured data store for typer.ctx."""

    project: DeetProject | None = None


P = ParamSpec("P")
R = TypeVar("R")


def project_required(f: Callable[P, R]) -> Callable[P, R]:
    """
    Check typer context for existence of a project, and exit if no project exists.

    This is used to decorate cli commands which we want to exit gracefully
    when they are run outside of a project directory.
    """

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        raw_ctx = kwargs.get("ctx")

        ctx = cast(typer.Context | None, raw_ctx)

        if not ctx or not ctx.obj or not ctx.obj.project:
            no_project = (
                "This command must be run from a directory that contains a project"
                " create one by running `deet project init`"
            )
            fail_with_message(no_project)
            raise typer.Exit(1)
        return f(*args, **kwargs)

    return wrapper
