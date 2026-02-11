"""A CLI app to run DEET pipelines."""

from pathlib import Path

import typer
from loguru import logger

from deet.data_models.base import SupportedImportFormat

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
    logger.info("Testing LLM Config")


def main() -> None:
    """Run CLI app."""
    app()


if __name__ == "__main__":
    app()
