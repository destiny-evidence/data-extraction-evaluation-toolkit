# ruff: noqa: PLC0415
"""A CLI app to run deet pipelines."""

import warnings

import typer

from deet.logger import logger
from deet.scripts.commands import project, run

APP_HELP = (
    "deet (data extraction evaluation toolkit) 🚤\n\n"
    "Use the deet CLI to extract data from documents with LLMs, and evaluate "
    "extraction by comparing to human-annotated data. To run any of the list "
    "of commands below, type `deet *command*`, and type `deet *command* --help` "
    "to see more information about the command. For example, `deet extract-data "
    "--help` \n"
    "Prefix any command with --verbose to see complete log output."
    "will give you more information about how to use the extract-data command.\n\n"
    "Run `deet --install-completion` to enable your shell to autocomplete deet "
    "commands."
)

app = typer.Typer(help=APP_HELP, add_completion=True)

app.add_typer(project.app, name="project")
app.add_typer(run.app, name="run")


@app.callback()
def global_options(
    *, verbose: bool = typer.Option(default=False, help="Display verbose logs.")
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


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
