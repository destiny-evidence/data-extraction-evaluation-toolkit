# ruff: noqa: PLC0415
"""A CLI app to run deet pipelines."""

import contextlib
import warnings

import typer

from deet.data_models.project import DeetProject
from deet.logger import logger
from deet.scripts.commands import project, run
from deet.scripts.context import CLIState
from deet.ui.terminal import console, render_template
from deet.ui.terminal.components import info_panel
from deet.ui.terminal.templates import APP_HELP

app = typer.Typer(help=APP_HELP, add_completion=True, rich_markup_mode="rich")

app.add_typer(project.app, name="project")
app.add_typer(run.app, name="run")


@app.callback(invoke_without_command=True)
def global_options(
    ctx: typer.Context,
    *,
    verbose: bool = typer.Option(default=False, help="Display verbose logs."),
) -> None:
    """Set global options for all deet commands."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(
        typer.echo,
        colorize=True,
        level=log_level,
        filter=lambda record: "is_echo" not in record["extra"],
    )
    if not verbose:
        warnings.filterwarnings("ignore", message=".*is ill-defined.*")

    cli_state = CLIState()

    with contextlib.suppress(FileNotFoundError):
        cli_state.project = DeetProject.load()

    ctx.obj = cli_state

    if ctx.invoked_subcommand is None:
        md_text = render_template("welcome", project=cli_state.project)
        console.clear()
        console.print(info_panel(md_text))


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
