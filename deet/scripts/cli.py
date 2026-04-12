# ruff: noqa: PLC0415
"""A CLI app to run deet pipelines."""

import contextlib
import warnings

import typer

from deet.logger import logger
from deet.scripts.commands import experiments, project
from deet.scripts.commands.deprecated import (
    export_config_template_legacy,
    extract_data_legacy,
    init_linkage_mapping_file_legacy,
    init_prompt_csv_legacy,
    link_documents_fulltexts_legacy,
    test_llm_config_legacy,
)
from deet.scripts.typer_context import CLIState
from deet.ui.terminal import console, render_template
from deet.ui.terminal.components import info_panel
from deet.ui.terminal.templates import APP_HELP

app = typer.Typer(help=APP_HELP, add_completion=True, rich_markup_mode="rich")

app.add_typer(project.app, name="project")
app.add_typer(experiments.app, name="experiments")

# Legacy commands - these just return instructions on how to use the new CLI
app.command(name="export-config-template", hidden=True)(export_config_template_legacy)
app.command(name="extract-data", hidden=True)(extract_data_legacy)
app.command(name="init-linkage-mapping-file", hidden=True)(
    init_linkage_mapping_file_legacy
)
app.command(name="init-prompt-csv", hidden=True)(init_prompt_csv_legacy)
app.command(name="link-documents-fulltexts", hidden=True)(
    link_documents_fulltexts_legacy
)
app.command(name="test-llm-config", hidden=True)(test_llm_config_legacy)


@app.callback(invoke_without_command=True)
def global_options(
    typer_context: typer.Context,
    *,
    verbose: bool = typer.Option(default=False, help="Display verbose logs."),
) -> None:
    """Set global options for all deet commands."""
    from deet.data_models.project import DeetProject

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

    typer_context.obj = cli_state

    if typer_context.invoked_subcommand is None:
        md_text = render_template("welcome", project=cli_state.project)
        console.clear()
        console.print(info_panel(md_text))


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
