# ruff: noqa: PLC0415
"""A CLI app to run deet pipelines."""

import warnings
from pathlib import Path
from typing import Annotated

import typer

from deet.data_models.enums import CustomPromptPopulationMethod
from deet.logger import logger
from deet.processors.converter_register import SupportedImportFormat
from deet.utils.cli_utils import echo_and_log, fail_with_message

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
    processed_annotation_data.export_linkage_mapper_csv(link_map_path)


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

    """
    from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy

    converter = gs_data_format.get_annotation_converter()
    processed_annotation_data = converter.process_annotation_file(gs_data_path)

    linker = DocumentReferenceLinker(
        references=processed_annotation_data.documents,
        document_base_dir=pdf_dir,
        document_reference_mapping=link_map_path,
        linking_strategies=[LinkingStrategy.MAPPING_FILE],
    )
    linked_documents = linker.link_many_references_parsed_documents()

    if not output_path.exists():
        output_path.mkdir()

    if len(linked_documents) == 0:
        fail_with_message("Error. Could not link any documents!")

    for linked_document in linked_documents:
        file_path = output_path / f"{linked_document.safe_identity.document_id}.json"
        linked_document.save(file_path)


@app.command()
def init_prompt_csv(
    gs_data_path: GS_DATA_PATH,
    gs_data_format: GS_DATA_FORMAT = DEFAULT_IMPORT_FORMAT,
    csv_path: Annotated[
        Path, typer.Option(help="A path to a file to write your prompt definitions to.")
    ] = DEFAULT_PROMPT_DEFINITION_PATH,
) -> None:
    """
    Write a csv to define prompts for your dataset with.

    This writes a row for each attribute in your dataset. Edit the prompt
    column to edit the prompt to be used for that attribute. Attributes
    without values in the prompt column will not be extracted.
    """
    converter = gs_data_format.get_annotation_converter()
    processed_annotation_data = converter.process_annotation_file(gs_data_path)
    if csv_path.exists():
        message = (
            "Prompt definition csv already exists. Proceeding will "
            "overwrite this and you may lose work. Do you want to continue?"
        )
        proceed = typer.confirm(message)
        if proceed:
            echo_and_log("Proceeding to overwrite prompt definition csv")
            csv_path.unlink()
        else:
            raise typer.Abort()  # noqa: RSE102
    processed_annotation_data.export_attributes_csv_file(filepath=csv_path)


@app.command()
def extract_data(  # noqa: PLR0913
    gs_data_path: GS_DATA_PATH,
    config_path: Annotated[
        Path,
        typer.Option(
            help="A path to a config file containing options for data "
            "extraction config. A template can be generated by running "
            "`deet export-config-template."
        ),
    ] = DEFAULT_CONFIG_PATH,
    gs_data_format: GS_DATA_FORMAT = DEFAULT_IMPORT_FORMAT,
    prompt_population: Annotated[
        CustomPromptPopulationMethod | None,
        typer.Option(
            help="A method to define custom prompts for your attributes to be "
            "extracted. Leave blank to use the prompts in your gold standard "
            "data. Set to `file` to provide a file of prompt definitions "
            "(make sure this is supplied below). Set to `cli` to define prompts"
            " interactively in the CLI."
        ),
    ] = None,
    csv_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to read custom prompt definitions from."
            " This must be set if using prompt population from file."
        ),
    ] = None,
    linked_document_path: Annotated[
        Path,
        typer.Option(
            help="A path to a directory containing documents that have been "
            "linked to their fulltexts. This directory can be populated by "
            "running `deet link-documents-fulltexts`."
        ),
    ] = DEFAULT_LINKED_DOCUMENTS_PATH,
    link_map_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to a link map (create this by running "
            "`deet init-linkage-mapping-file`). You must specify"
            "either a link map or a directory containing linked"
            "documents"
        ),
    ] = None,
    pdf_dir: Annotated[
        Path,
        typer.Option(
            help="Path to a directory containing pdfs. We will attempt to link"
            " these by document ID."
        ),
    ] = DEFAULT_PDF_PATH,
    out_dir: Annotated[
        Path,
        typer.Option(
            help="A path to a directory where you want to store the results of"
            " this, and further instances of extract-data for this project."
        ),
    ] = DEFAULT_EXPERIMENT_OUT_DIR,
    run_name: Annotated[
        str,
        typer.Option(
            help="A name for the run (which will appended to a timestamp) "
            "to help you identify this run later"
        ),
    ] = "",
    custom_evaluation_metrics: Annotated[
        list[str] | None,
        typer.Option(
            help="A list of additional sklearn metrics that you wish to "
            " calculate. Use this option for each additional metric you "
            " would like to add, e.g. `deet extract-data "
            "--custom-evaluation-metrics brier_score_loss "
            "--custom-evaluation-metrics cohen_kappa_score`"
        ),
    ] = None,
) -> None:
    """
    Extract data from documents and evaluate.

    Load gold standard annotation data, and use an LLM to extract data from the
    documents in your dataset. Evaluate by comparing the results to the gold
    standard data.
    """
    import yaml

    from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
    from deet.extractors.cli_helpers import (
        init_extraction_run,
        load_or_init_config,
        prepare_documents,
    )
    from deet.extractors.llm_data_extractor import LLMDataExtractor

    config = load_or_init_config(config_path=config_path)

    extraction_run_id, experiment_out_dir = init_extraction_run(out_dir, run_name)

    converter = gs_data_format.get_annotation_converter()
    processed_annotation_data = converter.process_annotation_file(gs_data_path)

    if prompt_population == CustomPromptPopulationMethod.FILE and not csv_path:
        message = "CSV prompt population selected without specifying csv_path"
        fail_with_message(message)
    if prompt_population is not None:
        processed_annotation_data.populate_custom_prompts(
            method=prompt_population, filepath=csv_path
        )

    data_extractor = LLMDataExtractor(config=config)

    documents = prepare_documents(
        processed_annotation_data.documents,
        config,
        linked_document_path=linked_document_path,
        pdf_dir=pdf_dir,
        link_map_path=link_map_path,
    )

    llm_annotated_documents = data_extractor.extract_from_documents(
        attributes=processed_annotation_data.attributes,
        documents=documents,
        context_type=data_extractor.config.default_context_type,
        output_file=experiment_out_dir / "annotated_docs.json",
        show_progress=True,
    )

    config_out = experiment_out_dir / "config.yaml"
    config_out.write_text(
        yaml.safe_dump(data_extractor.config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_annotation_data.annotated_documents,
        llm_annotated_documents=llm_annotated_documents,
        attributes=processed_annotation_data.attributes,
        custom_metrics=custom_evaluation_metrics,
        extraction_run_id=extraction_run_id,
    )
    evaluator.evaluate_llm_annotations()
    evaluator.write_metrics_to_csv(experiment_out_dir / DEFAULT_METRICS_CSV)
    evaluator.export_llm_comparison(experiment_out_dir / DEFAULT_OUTPUT_COMPARISON_CSV)
    evaluator.display_metrics()


@app.command()
def test_llm_config() -> None:
    """Test llm config."""
    from deet.data_models.base import Attribute, AttributeType
    from deet.extractors.llm_data_extractor import (
        DataExtractionConfig,
        LLMDataExtractor,
    )

    config = DataExtractionConfig()
    data_extractor = LLMDataExtractor(config=config)
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Is the document about climate and health? Return a BOOL",
    )
    context = (
        "This is document, extract data from me please. I am about climate and health"
    )
    response = data_extractor.extract_from_document(
        attributes=[attr],
        payload=context,
        context_type=None,
    )
    echo_and_log(response)


@app.callback()
def global_options(
    *, verbose: bool = typer.Option(default=False, help="Display verbose logs.")
) -> None:
    """Set global options for all deet commands."""
    level = "DEBUG" if verbose else "INFO"
    logger.add(typer.echo, colorize=True, level=level)
    if not verbose:
        warnings.filterwarnings("ignore", message=".*is ill-defined.*")


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
