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

app = typer.Typer(help=APP_HELP, add_completion=True)

# Shared argument definitions and defaults
DEFAULT_CONFIG_PATH = Path("default_extraction_config.yaml")

GS_DATA_PATH = Annotated[
    Path,
    typer.Argument(..., help="Path to gold standard annotation file."),
]

DEFAULT_IMPORT_FORMAT = SupportedImportFormat.EPPI_JSON

GS_DATA_FORMAT = Annotated[
    SupportedImportFormat,
    typer.Option(
        help="Format of the input data (determines which converter to use)",
    ),
]

DEFAULT_LINK_MAP = Path("link_map.csv")

LINK_MAP_PATH = Annotated[
    Path,
    typer.Option(
        help="Path to write the link map",
    ),
]

LINK_MAP_PATH_READ = Annotated[
    Path,
    typer.Option(
        help="A path to a link map (create this by running "
        "`deet init-linkage-mapping-file`)"
    ),
]

DEFAULT_PDF_PATH = Path("pdfs")

DEFAULT_PROMPT_DEFINITION_PATH = Path("prompt_definitions.csv")

DEFAULT_EXPERIMENT_OUT_DIR = Path("data-extraction-experiments/")
DEFAULT_METRICS_CSV = Path("metrics.csv")
DEFAULT_OUTPUT_COMPARISON_CSV = Path("goldstandard_llm_comparison.csv")

DEFAULT_LINKED_DOCUMENTS_PATH = Path("linked_documents")


@app.command()
def export_config_template(
    output_path: Annotated[
        Path,
        typer.Option(help="The output path where your config file will be written"),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    """Export the default DataExtractionConfig to a YAML file."""
    import yaml  # type:ignore[import-untyped]

    from deet.extractors.llm_data_extractor import DataExtractionConfig

    config = DataExtractionConfig()
    if output_path.exists():
        message = (
            "Config template exists. Proceeding will "
            "overwrite this and you may lose work if you have edited this."
            " Do you want to continue?"
        )
        proceed = typer.confirm(message)
        if proceed:
            echo_and_log("Proceeding to overwrite config template")
            output_path.unlink()
        else:
            raise typer.Abort()  # noqa: RSE102
    output_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    echo_and_log(f"✅ Default config exported to {output_path}", fg=typer.colors.GREEN)
    echo_and_log(
        "✏️  Edit this file to adjust options for data extraction.", fg=typer.colors.BLUE
    )


@app.command()
def init_linkage_mapping_file(
    gs_data_path: GS_DATA_PATH,
    gs_data_format: GS_DATA_FORMAT = DEFAULT_IMPORT_FORMAT,
    link_map_path: LINK_MAP_PATH = DEFAULT_LINK_MAP,
    pdf_dir: Annotated[
        Path | None,
        typer.Option(
            help="Optional directory of pdfs/mds. If provided, deet will attempt "
            "to pre-fill the file_path column using available linking strategies "
            "(filename ID match, then author-year match)."
        ),
    ] = None,
) -> None:
    """Create a mapping to link documents and their full texts."""
    if link_map_path.exists():
        message = (
            f"mapping already exists at {link_map_path}. Overwriting"
            " may cause you to lose work. Do you want to continue?"
        )
        proceed = typer.confirm(message)
        if proceed:
            echo_and_log("Proceeding to overwrite config template")
            link_map_path.unlink()
        else:
            raise typer.Abort()  # noqa: RSE102

    converter = gs_data_format.get_annotation_converter()
    processed_annotation_data = converter.process_annotation_file(gs_data_path)
    processed_annotation_data.export_linkage_mapper_csv(
        link_map_path,
        document_base_dir=pdf_dir,
    )


@app.command()
def link_documents_fulltexts(
    gs_data_path: GS_DATA_PATH,
    link_map_path: LINK_MAP_PATH_READ = DEFAULT_LINK_MAP,
    gs_data_format: GS_DATA_FORMAT = DEFAULT_IMPORT_FORMAT,
    pdf_dir: Annotated[
        Path, typer.Option(help="Path to a directory containing pdfs.")
    ] = DEFAULT_PDF_PATH,
    output_path: Annotated[
        Path,
        typer.Option(help="A path to a directory to write the linked documents to."),
    ] = DEFAULT_LINKED_DOCUMENTS_PATH,
) -> None:
    """
    Link documents to their fulltexts.

    This creates a document containing the parsed output of its corresponding
    fulltext in the folder defined in `output_path`. Linking will be
    attempted using a mapping file, if provided, then by matching the
    filename with author and year, then by matching by document id. See
    `deet.processors.linker` for more details.
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
