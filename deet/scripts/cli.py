"""A CLI app to run DEET pipelines."""

import csv
import datetime
from pathlib import Path

import typer
import yaml

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.documents import (
    Document,
)
from deet.data_models.processed_gold_standard_annotations import (
    BaseProcessedAnnotationData,
    CustomPromptPopulationMethod,
    ProcessedEppiAnnotationData,
)
from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
from deet.extractors.llm_data_extractor import (
    ContextType,
    DataExtractionConfig,
    LLMDataExtractor,
)
from deet.logger import logger
from deet.processors.base_converter import Outfiles
from deet.processors.converter_register import SupportedImportFormat
from deet.processors.linker import DocumentReferenceLinker
from deet.settings import get_settings

settings = get_settings()

app = typer.Typer()


@app.command()
def export_config_template(
    output_path: Path = Path("default_extraction_config.yaml"),
) -> None:
    """Export the default DataExtractionConfig to a YAML file."""
    config = DataExtractionConfig()
    output_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    typer.echo(f"Default config exported to {output_path}")


@app.command()
def import_gold_standard_data(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.EPPI_JSON,
    output_dir: Path | None = Path(),
) -> BaseProcessedAnnotationData | ProcessedEppiAnnotationData:
    """
    Import gold standard annotation data from a supported format.

    Args:
        gs_data_path (Path): A path to a file or directory containing gold
            standard annotations.
        gs_data_format (SupportedImportFormat): Format of the input data.
            This determines which converter to use. Defaults to
            SupportedImportFormat.EPPI_JSON.
        output_dir (Path): Directory where processed data files will be
            written. Set to None if you do not want to write processed data
            to disk. Defaults to the current working directory.

    Returns:
        ProcessedAnnotationData: A structured object containing the
            parsed annotation data.


    """
    converter = gs_data_format.get_annotation_converter()
    if gs_data_path.is_dir():
        out = converter.reload_output(gs_data_path)
    else:
        out = converter.process_annotation_file(gs_data_path)

    logger.info(f"Imported data: {len(out.annotated_documents)} annotated documents.")
    if output_dir is not None:
        converter.write_processed_data_to_file(
            processed_data=out, output_dir=output_dir, outfiles_to_write=list(Outfiles)
        )
    return out


@app.command()
def init_linkage_mapping_file(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.EPPI_JSON,
    link_map_path: Path = Path("link_map.csv"),
) -> None:
    """
    Create a mapping to link documents and the full texts.

    Args:
        gs_data_path (Path): A path to a file or directory containing gold
            standard annotations.
        gs_data_format (SupportedImportFormat): Format of the input data.
            This determines which converter to use. Defaults to
            SupportedImportFormat.EPPI_JSON.
        link_map_path (Path): A path to write the link_map template to.
            Defaults to link_map.csv in the current working directory.

    """
    out = import_gold_standard_data(
        gs_data_path=gs_data_path, gs_data_format=gs_data_format
    )

    with link_map_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "name", "file_path"])
        writer.writeheader()
        for d in out.documents:
            d.init_document_identity()
            if d.document_identity is None:
                message = f"document_identity was not set for document {d}"
                raise RuntimeError(message)
            writer.writerow(
                {
                    "document_id": d.document_identity.document_id,
                    "name": d.name,
                    "file_path": None,
                }
            )


@app.command()
def link_documents_fulltexts(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.EPPI_JSON,
    pdf_dir: Path = Path("pdfs"),
    link_map_path: Path | None = None,
    output_path: Path = Path("linked_documents"),
) -> None:
    """
    Link documents to their fulltexts.

    This creates a document containing the parsed output of its corresponding
        fulltext in the folder defined in `output_path`. Linking will be
        attempted using a mapping file, if provided, then by matching the
        filename with author and year, then by matching by document id. See
        `deet.processors.linker` for more details.

    Args:
        gs_data_path (Path): A path to a file or directory containing gold
            standard annotations.
        gs_data_format (SupportedImportFormat): Format of the input data.
            This determines which converter to use. Defaults to
            SupportedImportFormat.EPPI_JSON.
        pdf_dir (Path): A path to a directory containing pdfs. Defaults to
            "pdfs", within the current directory.
        link_map_path (Path): A path to a link map
            (create this by running `deet init-linkage-mapping-file`).
        output_path (Path): A path to a directory to write the linked documents
            to. This defaults to `linked_documents`.

    """
    out = import_gold_standard_data(
        gs_data_path=gs_data_path, gs_data_format=gs_data_format
    )

    linker = DocumentReferenceLinker(
        references=out.documents,
        document_base_dir=pdf_dir,
        document_reference_mapping=link_map_path,
    )
    linked_documents = linker.link_many_references_parsed_documents()
    for linked_document in linked_documents:
        file_path = output_path / f"{linked_document.safe_identity.document_id}.json"
        linked_document.save(file_path)


@app.command()
def init_prompt_csv(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.EPPI_JSON,
    csv_path: Path = Path("prompt_definitions.csv"),
) -> None:
    """
    Write a csv to define prompts for your dataset with.

    This writes a row for each attribute in your dataset. Edit the prompt
        column to edit the prompt to be used for that attribute. Attributes
        without values in the prompt column will not be extracted.

    Args:
        gs_data_path (Path): A path to a file or directory containing gold
            standard annotations.
        gs_data_format (SupportedImportFormat): Format of the input data.
            This determines which converter to use. Defaults to
            SupportedImportFormat.EPPI_JSON.
        csv_path (Path): a path to a file to write your prompt definitions to.
            Defaults to `prompt_defitions.csv` in the current working directory.

    """
    out = import_gold_standard_data(
        gs_data_path=gs_data_path, gs_data_format=gs_data_format
    )
    if csv_path.exists():
        message = (
            "Prompt definition csv already exists. Proceeding will "
            "overwrite this and you may lose work. Do you want to continue?"
        )
        proceed = typer.confirm(message)
        if proceed:
            logger.info("Proceeding to overwrite prompt definition csv")
            csv_path.unlink()
        else:
            raise typer.Abort()  # noqa: RSE102
    out.export_attributes_csv_file(filepath=csv_path)


@app.command()
def extract_data(  # noqa: PLR0913
    config_path: Path = Path("default_extraction_config.yaml"),
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.EPPI_JSON,
    prompt_population: CustomPromptPopulationMethod | None = None,
    csv_path: Path | None = None,
    linked_document_path: Path = Path("linked_documents"),
    out_dir: Path | None = None,
    run_name: str = "",
) -> None:
    """
    Extract data from documents.

    Load gold standard annotation data, and use an LLM to extract data from the
        documents in your dataset.

    Args:
        config_path (Path): A path to a config file containing options for
            data extraction config. A template can be generated by running
            `deet export-config-template`
        gs_data_path (Path): A path to a file or directory containing gold
            standard annotations.
        gs_data_format (SupportedImportFormat): Format of the input data.
            This determines which converter to use. Defaults to
            SupportedImportFormat.EPPI_JSON.
        prompt_population (CustomPromptPopulationMethod): A method to define
            custom prompts for your attributes to be extracted. Leave blank
            to use the prompts in your gold standard data. Set to `file` to
            provide a file of prompt definitions (make sure this is supplied
            below). Set to `cli` to define prompts interactively in the CLI.
        csv_path (Path): A path to a file to write your prompt definitions to.
            Defaults to `prompt_defitions.csv` in the current working directory.
        linked_document_path (Path): A path to a directory containing
            documents that have been linked to their fulltexts. This directory
            can be populated by running `deet link-documents-fulltexts`
        out_dir (Path): A path to a directory where you want to store the
            results of this, and further instances of extract-data for this
            project. Defaults to the current directory.
        run_name (Path): A name for the run (which will appended to a timestamp)
            to help you identify this run later

    """
    if config_path.exists():
        config = DataExtractionConfig(**yaml.safe_load(config_path.read_text()))
    else:
        logger.warning(
            f"Config file: {config_path} does not exist."
            " Initialising config with default settings."
        )
        config = DataExtractionConfig()

    pipeline_run_id = (
        datetime.datetime.now(tz=datetime.UTC).strftime("%Y/%m/%d, %H:%M:%S")
        + f"_{run_name}"
    )
    if out_dir is None:
        out_dir = Path("pipeline_runs") / pipeline_run_id
        out_dir.mkdir(parents=True)
    elif out_dir.exists():
        out_dir_exists = "out_dir already exists. Exiting, so as not to overwrite data"
        raise typer.Abort(out_dir_exists)

    out = import_gold_standard_data(
        gs_data_path=gs_data_path, gs_data_format=gs_data_format
    )

    if prompt_population == CustomPromptPopulationMethod.FILE and not csv_path:
        message = "CSV prompt popluation selected without specifying csv_path"
        raise ValueError(message)
    if prompt_population is not None:
        out.populate_custom_prompts(method=prompt_population, filepath=csv_path)

    data_extractor = LLMDataExtractor(config=config)

    if config.default_context_type == ContextType.ABSTRACT_ONLY:
        documents = out.documents
    elif config.default_context_type == ContextType.FULL_DOCUMENT:
        if linked_document_path.exists():
            documents = [Document.load(f) for f in linked_document_path.glob("*.json")]
        else:
            message = f"context type {config.default_context_type} selected"
            " but no linked_document_path supplied"
            raise typer.Abort(message)
    else:
        message = f"context type {config.default_context_type} not supported"
        raise typer.Abort(message)

    llm_annotated_documents = data_extractor.extract_from_documents(
        attributes=out.attributes,
        documents=documents,
        context_type=data_extractor.config.default_context_type,
        output_file=out_dir / "annotated_docs.json",
    )

    config_out = out_dir / "config.yaml"
    config_out.write_text(
        yaml.safe_dump(data_extractor.config.model_dump(mode="json"), sort_keys=False)
    )

    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=out.annotated_documents,
        llm_annotated_documents=llm_annotated_documents,
        attributes=out.attributes,
    )
    evaluator.evaluate_llm_annotations()
    evaluator.display_metrics()


@app.command()
def test_llm_config() -> None:
    """Test llm config."""
    config = DataExtractionConfig()
    data_extractor = LLMDataExtractor(config=config)
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Is the document about climate and health? Return a BOOL",
    )
    context = (
        "This is document, extract data from me please. "
        "I am about climate and health"
    )
    response = data_extractor.extract_from_document(
        attributes=[attr],
        payload=context,
        context_type=None,
    )
    logger.info(response)


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
