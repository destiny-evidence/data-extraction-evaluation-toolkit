"""A CLI app to run DEET pipelines."""

from pathlib import Path

import typer

from deet.data_models.base import Attribute, AttributeType
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
from deet.logger import logger
from deet.processors.base_converter import SupportedImportFormat
from deet.settings import get_settings

settings = get_settings()

app = typer.Typer()


@app.command()
def import_data(gs_data_path: Path, gs_data_format: SupportedImportFormat) -> None:
    """Import gold standard annotation data from a supported format."""
    converter = gs_data_format.get_annotation_converter()
    out = converter.process_annotation_file(gs_data_path)
    logger.info(f"Imported data: {len(out.annotated_documents)} annotated documents.")


@app.command()
def test_llmconfig() -> None:
    """Test llm config."""
    config = DataExtractionConfig()
    data_extractor = LLMDataExtractor(config=config)
    attr = Attribute(
        question_target="Test question",
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
