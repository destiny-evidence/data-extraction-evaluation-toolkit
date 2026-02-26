"""A CLI app to run DEET pipelines."""

import csv
from pathlib import Path

import typer
import yaml
from destiny_sdk.enhancements import EnhancementType
from ftfy import fix_text
from uuid6 import uuid7

from deet.data_models.base import (
    Attribute,
    AttributeType,
)
from deet.data_models.documents import (
    Document,
    LinkedDocument,
)
from deet.data_models.processed import (
    BaseProcessedAnnotationData,
    ProcessedEppiAnnotationData,
    PromptPopulationMethod,
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

settings = get_settings()

app = typer.Typer()


def get_abstract(document: Document) -> str:
    """Return the abstract of a reference. Will be replaced by feature/linking."""
    for e in document.citation.enhancements or []:
        if e.content.enhancement_type == EnhancementType.ABSTRACT:
            return fix_text(e.content.abstract)
    return ""


@app.command()
def export_default_config(
    output_path: Path = Path("default_extraction_config.yaml"),
) -> None:
    """Export the default DataExtractionConfig to a YAML file."""
    config = DataExtractionConfig()
    output_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    )
    typer.echo(f"Default config exported to {output_path}")


@app.command()
def import_data(
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
def create_link_map(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    link_map_path: Path = Path("link_map.csv"),
) -> None:
    """Create a mapping to link documents and the full texts."""
    out = import_data(gs_data_path=gs_data_path, gs_data_format=gs_data_format)

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
def link_documents(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    pdf_dir: Path = Path("pdfs"),
    link_map_path: Path | None = None,
    output_path: Path = Path("linked_documents"),
) -> None:
    """Link documents to their fulltexts."""
    out = import_data(gs_data_path=gs_data_path, gs_data_format=gs_data_format)

    linker = DocumentReferenceLinker(
        references=out.documents,
        document_base_dir=pdf_dir,
        document_reference_mapping=link_map_path,
    )
    linked_documents = linker.link_many_references_parsed_documents()
    for linked_document in linked_documents:
        file_path = (
            output_path / f"{linked_document.document_identity.document_id}.json"
        )
        linked_document.save(file_path)


@app.command()
def write_prompt_csv(
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    csv_path: Path = Path("prompt_definitions.csv"),
) -> None:
    """Write a prompt csv."""
    out = import_data(gs_data_path=gs_data_path, gs_data_format=gs_data_format)
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
def data_extraction(  # noqa: PLR0913
    config_path: Path = Path("default_extraction_config.yaml"),
    gs_data_path: Path = Path(),
    gs_data_format: SupportedImportFormat = SupportedImportFormat.DEET,
    prompt_population: PromptPopulationMethod = PromptPopulationMethod.ATTRIBUTEFILE,
    csv_path: Path | None = None,
    linked_document_path: Path = Path("linked_documents"),
    out_dir: Path | None = None,
) -> None:
    """Extract data from documents."""
    config = DataExtractionConfig(**yaml.safe_load(config_path.read_text()))

    if out_dir is None:
        out_dir = Path("pipeline_runs") / str(uuid7())
        out_dir.mkdir(parents=True)
    elif out_dir.exists():
        out_dir_exists = "out_dir already exists. Exiting, so as not to overwrite data"
        raise typer.Abort(out_dir_exists)

    out = import_data(gs_data_path=gs_data_path, gs_data_format=gs_data_format)

    if prompt_population == PromptPopulationMethod.FILE and not csv_path:
        message = "CSV prompt popluation selected without specifying csv_path"
        raise ValueError(message)
    out.populate_custom_prompts(method=prompt_population, filepath=csv_path)

    data_extractor = LLMDataExtractor(config=config)

    if config.default_context_type == ContextType.ABSTRACT_ONLY:
        documents = out.documents
    elif config.default_context_type == ContextType.FULL_DOCUMENT:
        if linked_document_path.exists():
            documents = [
                LinkedDocument.load(f) for f in linked_document_path.glob("*.json")
            ]
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


@app.command()
def test_llmconfig() -> None:
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
