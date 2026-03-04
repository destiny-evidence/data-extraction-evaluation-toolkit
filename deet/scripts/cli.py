"""A CLI app to run DEET pipelines."""

import csv
from pathlib import Path
from uuid import UUID

import typer
import yaml
from pydantic import TypeAdapter
from uuid6 import uuid7

from deet.data_models.base import Attribute, AttributeType, GoldStandardAnnotation
from deet.data_models.documents import (
    Document,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotatedDocumentList,
)
from deet.data_models.processed_gold_standard_annotations import (
    BaseProcessedAnnotationData,
    CustomPromptPopulationMethod,
    ProcessedEppiAnnotationData,
)
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
from deet.utils.cli_utils import get_last_pipeline_run
from deet.utils.evaluation_utils import display_metrics, evaluate_llm_annotations

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
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    output_dir: Path = Path(),
) -> BaseProcessedAnnotationData | ProcessedEppiAnnotationData:
    """
    Import gold standard annotation data from a supported format.

    Args:
        gs_data_path (Path): A path to a file or directory containing gold
        standard annotations
        gs_data_format (SupportedImportFormat): Format of the input data.
        This determines which converter to use. Defaults to SupportedImportFormat.DEET
        output_dir (Path): Directory where processed data files will be
        written if the input is not DEET. Defaults to the current working
        directory.

    Returns:
        ProcessedAnnotationData: A structured object containing the
        parsed annotation data.

    Notes:
        - If `gs_data_format` is DEET, the processed data will not be written to
          disk (as we assume we are just reading in data that has already been
          converted from another format and written to disk.)

    """
    converter = gs_data_format.get_annotation_converter()
    out = converter.process_annotation_file(gs_data_path)
    logger.info(f"Imported data: {len(out.annotated_documents)} annotated documents.")
    if gs_data_format != SupportedImportFormat.DEET:
        converter.write_processed_data_to_file(
            processed_data=out, output_dir=output_dir, outfiles_to_write=list(Outfiles)
        )
    return out


@app.command()
def init_linkage_mapping_file(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    link_map_path: Path = Path("link_map.csv"),
) -> None:
    """Create a mapping to link documents and the full texts."""
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
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    pdf_dir: Path = Path("pdfs"),
    link_map_path: Path | None = None,
    output_path: Path = Path("linked_documents"),
) -> None:
    """Link documents to their fulltexts."""
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
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    csv_path: Path = Path("prompt_definitions.csv"),
) -> None:
    """Write a prompt csv."""
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
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    prompt_population: CustomPromptPopulationMethod | None = None,
    csv_path: Path | None = None,
    linked_document_path: Path = Path("linked_documents"),
    out_dir: Path | None = None,
) -> None:
    """Extract data from documents."""
    if config_path.exists():
        config = DataExtractionConfig(**yaml.safe_load(config_path.read_text()))
    else:
        logger.warning(
            f"Config file: {config_path} does not exist."
            " Initialising config with default settings."
        )
        config = DataExtractionConfig()

    pipeline_run_id = uuid7()
    if out_dir is None:
        out_dir = Path("pipeline_runs") / str(pipeline_run_id)
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

    data_extractor.extract_from_documents(
        attributes=out.attributes,
        documents=documents,
        context_type=data_extractor.config.default_context_type,
        output_file=out_dir / "annotated_docs.json",
    )

    config_out = out_dir / "config.yaml"
    config_out.write_text(
        yaml.safe_dump(data_extractor.config.model_dump(mode="json"), sort_keys=False)
    )

    evaluate_llm_to_gs(gs_data_path, gs_data_format, pipeline_run_id)


@app.command()
def evaluate_llm_to_gs(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    pipeline: UUID | None = None,
) -> None:
    """Evaluate a pipeline run, and print a table of evaluation metrics."""
    out = import_gold_standard_data(
        gs_data_path=gs_data_path, gs_data_format=gs_data_format
    )

    if pipeline is None:
        pipeline, pipeline_dir = get_last_pipeline_run(Path("pipeline_runs"))
    else:
        pipeline_dir = Path("pipeline_runs") / str(pipeline)

    adapter = TypeAdapter(
        list[GoldStandardAnnotatedDocument[Document, GoldStandardAnnotation]]
    )
    llm_annotation_file = pipeline_dir / "annotated_docs.json"
    llm_annotation_list = GoldStandardAnnotatedDocumentList(
        gold_standard_annotations=adapter.validate_json(llm_annotation_file.read_text())
    )

    metrics = evaluate_llm_annotations(
        reference_documents=out.annotated_documents,
        attributes=out.attributes,
        llm_annotation_list=llm_annotation_list,
        pipeline_run_id=pipeline,
    )

    display_metrics(metrics)


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
