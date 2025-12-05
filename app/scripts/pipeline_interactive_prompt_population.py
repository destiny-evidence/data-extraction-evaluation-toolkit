"""
An example of a pipeline script where prompts are supplied
by the user, attribute by attribute. The user writes the
prompts interactively as a stage in the pipeline.

The logic for this is implemented in lines 91-93
of this file.
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from app.data_models.base import Attribute, Document, GoldStandardAnnotation

# @sagaruprety note that we now only use Eppi types in our
# specific use-case (i.e. a pipeline script), no longer in the
# underlying application. The application uses base.py data types.
from app.data_models.eppi import EppiAttribute, EppiDocument
from app.data_models.pipeline import JobType, Pipeline, jobify, stage_from_job
from app.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
from app.processors.eppi_annotation_converter import EppiAnnotationConverter
from app.processors.parser import DocumentParser

parser = DocumentParser()
converter = EppiAnnotationConverter()

# NOTE - define your LLM config stuff here. currently all values are default.
config = DataExtractionConfig()

data_extractor = LLMDataExtractor(config=config)


# the three functions we want to run
def parse_pdf(
    pdf_path: Path,
    out_path: Path,
    *,
    skip_if_md_exists: bool = True,
) -> None:
    """
    Parse pdf to Markdown.

    Args:
        pdf_path (Path): location of input pdf.
        out_path (Path): location to write markdown file to.
        skip_if_md_exists (bool, optional): set to true if you want to skip this stage
                                            if markdown already exists.
                                            NOTE: you are responsible for ensuring md
                                            file matches the pdf. Defaults to True.

    """
    if skip_if_md_exists and out_path.exists() and out_path.is_file():
        logger.info(
            f"`skip_if_md_exists` has been set to True, and {str(out_path)}  exists. "  # noqa: RUF010
            "skipping parsing..."
        )
        return
    parser(input_=pdf_path, out_path=out_path)


def ingest_gold_standard_func(eppi_json_path: Path, output_dir: Path) -> None:
    """Convert EPPI JSON to DEET data models."""
    out = converter.process_annotation_file(eppi_json_path)
    converter.save_processed_data(processed_data=out, output_dir=output_dir)


def llm_data_extraction(
    full_text_path: Path,
    documents_file_path: Path,
    attributes_file_path: Path,
    output_path: Path,
    filter_by_attribute_ids: list[int] | None = None,
    **kwargs,
) -> list[GoldStandardAnnotation]:
    """Run LLM data extraction."""
    full_text = full_text_path.read_text()

    documents_raw = json.loads(documents_file_path.read_text())
    attributes_raw = json.loads(attributes_file_path.read_text())

    attributes: list[Attribute] = [EppiAttribute(**record) for record in attributes_raw]
    if filter_by_attribute_ids:
        attributes = [
            a for a in attributes if a.attribute_id in filter_by_attribute_ids
        ]

    # add prompts interactively:
    for att in attributes:
        att.enter_custom_prompt()

    documents: list[Document] = [EppiDocument(**record) for record in documents_raw]

    return data_extractor.extract_from_documents(
        documents=documents,
        attributes=attributes,
        output_file=output_path,
        full_text=full_text,
        **kwargs,
    )


def main() -> None:
    """Run main part of script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pdf_path", help="incoming pdf file", required=False, type=Path
    )
    parser.add_argument(
        "-m",
        "--markdown_path",
        help="path to save markdown at",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-e", "--eppi_json_path", help="path to eppi json", type=Path, required=True
    )

    args = parser.parse_args()

    if not args.pdf_path:
        raise ValueError("pdf_path is required")
    eppi_json_dir = str(Path(args.eppi_json_path).name).split(".")[:-1][0]
    eppi_out_path = (
        Path(args.pdf_path).parent / "tmp_parsed_eppi" / eppi_json_dir / "eppi"
    )

    logger.debug(eppi_out_path)
    if not args.markdown_path:
        args.markdown_path = Path(str(args.pdf_path).split(".")[:-1][0] + ".md")

    # Create stages by wrapping the jobified functions
    logger.debug("decorating our functions as Jobs and PipelineStages")
    parse_pdf_stage = stage_from_job(
        jobify(
            name="parse_pdf",
            func_kwargs={
                "pdf_path": args.pdf_path,
                "out_path": args.markdown_path,
            },
        )(parse_pdf)  # Apply jobify decorator to function
    )

    ingest_gs_stage = stage_from_job(
        jobify(
            name="ingest_gs",
            func_kwargs={
                "eppi_json_path": args.eppi_json_path,
                "output_dir": eppi_out_path,
            },
        )(ingest_gold_standard_func)
    )

    llm_extraction_stage = stage_from_job(
        jobify(
            name="llm_extraction",
            job_type=JobType.EXTRACTION,
            func_kwargs={
                "full_text_path": args.markdown_path,
                "documents_file_path": eppi_out_path / "documents.json",
                "attributes_file_path": eppi_out_path / "attributes.json",
                "output_path": eppi_out_path / "llm_extractions.json",
                "prompt_outfile": eppi_out_path / "full_prompt_payload.json",
            },
        )(llm_data_extraction)
    )

    my_beautiful_pipeline = Pipeline(
        name="test_pipeline",
        stages=[parse_pdf_stage, ingest_gs_stage, llm_extraction_stage],
    )

    my_beautiful_pipeline.run()


if __name__ == "__main__":
    main()
