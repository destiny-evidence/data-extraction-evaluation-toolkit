"""
An example of a pipeline script where prompts are supplied
by the user, asynchronously.

This is a two-part pipeline.
This script is part 02, where the user-written prompts are
incorporated into the attributes that are used by the
LLM extractor to extract info.

Before running this, the user has to
- run pipeline_prompt_csv_01.py
- write prompts into the csv created by that pipeline
- make a note of that csv's location and pass it to
  this script with a -c flag.

"""

import argparse
import json
from pathlib import Path

from loguru import logger

from deet.data_models.base import Attribute, ContextType, GoldStandardAnnotation
from deet.data_models.eppi import EppiAttribute
from deet.data_models.pipeline import JobType, Pipeline, jobify, stage_from_job
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
from deet.processors.eppi_annotation_converter import (
    DEFAULT_ATTRIBUTES_FILENAME,
    DEFAULT_BASE_OUTPUT_DIR,
    EppiAnnotationConverter,
)
from deet.processors.parser import DocumentParser

parser = DocumentParser()
converter = EppiAnnotationConverter()

# NOTE - define your LLM config stuff here. currently all values are default.
config = DataExtractionConfig(selected_attribute_ids=[6080465, 6080480])

data_extractor = LLMDataExtractor(config=config)


# the three functions we want to run (one per pipeline stage)
def parse_pdf(
    pdf_path: Path,
    out_path: Path,
    *,
    skip_if_md_exists: bool = True,
) -> None:
    """
    Parse all PDFs in pdf_path (dir) to markdown files in out_path (dir).

    No-op if pdf_path or out_path is None (e.g. when only markdown dir is given).

    Args:
        pdf_path: Directory containing PDF files.
        out_path: Directory to write markdown files to.
        skip_if_md_exists: If True, skip parsing when markdown already exists.
            You are responsible for ensuring the md file matches the pdf.
            Defaults to True.

    """
    if pdf_path is None or out_path is None or not pdf_path.is_dir():
        missing_paths = "must specify a pdf_path and out_path"
        raise ValueError(missing_paths)

    out_path.mkdir(parents=True, exist_ok=True)
    pdf_files = [
        f for f in pdf_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"
    ]
    for single_pdf in sorted(pdf_files):
        md_path = out_path / f"{single_pdf.stem}.md"
        if skip_if_md_exists and md_path.exists() and md_path.is_file():
            logger.info(
                f"`skip_if_md_exists` has been set to True, and {md_path!s} "
                "exists. skipping parsing..."
            )
            continue
        parser(input_=single_pdf, out_path=md_path)


def ingest_gold_standard_func(
    eppi_json_path: Path, output_dir: Path, csv_path: Path
) -> None:
    """Convert EPPI JSON to DEET data models and import prompts from CSV."""
    out = converter.process_annotation_file(eppi_json_path)

    if csv_path.parent == Path("."):  # noqa: PTH201
        csv_path = output_dir / csv_path

    out.populate_custom_prompts(method="file", filepath=csv_path)
    converter.write_processed_data_to_file(processed_data=out, output_dir=output_dir)


def llm_data_extraction(  # noqa: PLR0913
    markdown_dir: Path,
    attributes_file_path: Path,
    output_path: Path,
    pdf_dir: Path | None = None,
    filter_by_attribute_ids: list[int] | None = None,
    prompt_outfile: Path | None = None,
) -> dict[str, list[GoldStandardAnnotation]]:
    """
    Run LLM data extraction for all files in the markdown dir.

    Delegates to LLMDataExtractor.extract_from_documents (loop over files
    inside the extractor).

    Args:
        markdown_dir: Directory of markdown files.
        attributes_file_path: Path to attributes JSON file.
        output_path: Path to save combined output JSON.
        pdf_dir: Directory of PDFs (optional); when set, lists inputs from here.
        filter_by_attribute_ids: Optional list of attribute IDs to filter.
        prompt_outfile: Optional path to write prompt payload for debugging.

    Returns:
        Dictionary mapping file paths to lists of annotations.

    """
    attributes_raw = json.loads(attributes_file_path.read_text(encoding="utf-8"))
    attributes: list[Attribute] = [EppiAttribute(**record) for record in attributes_raw]
    if filter_by_attribute_ids:
        attributes = [
            a for a in attributes if a.attribute_id in filter_by_attribute_ids
        ]

    return data_extractor.extract_from_documents(
        attributes=attributes,
        markdown_dir=markdown_dir,
        output_file=output_path,
        pdf_dir=pdf_dir,
        context_type=ContextType.FULL_DOCUMENT,
        prompt_outfile=prompt_outfile,
    )


def main() -> None:
    """Run main part of script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pdf_path",
        help="directory containing PDF files to process",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--markdown_path",
        help=(
            "directory containing markdown files "
            "(for checking existing markdowns or processing markdowns directly)"
        ),
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-e", "--eppi_json_path", help="path to eppi json", type=Path, required=True
    )
    parser.add_argument(
        "-c",
        "--csv_path",
        help="path you want to ingest manually populated pomrpt csv from.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help=("path to save output JSON (auto-generated if not provided)"),
        type=Path,
        required=False,
    )

    args = parser.parse_args()

    pdf_path = Path()
    markdown_path = Path()
    eppi_json_path: Path = args.eppi_json_path
    output_path: Path = args.output_path
    csv_path: Path = args.csv_path

    # Validate that at least one directory is provided
    if not args.pdf_path and not args.markdown_path:
        error_msg = (
            "At least one of -p/--pdf_path or -m/--markdown_path must be provided"
        )
        raise ValueError(error_msg)

    # Validate pdf_path if provided
    if args.pdf_path:
        if not args.pdf_path.exists():
            error_msg = f"PDF directory does not exist: {args.pdf_path}"
            raise ValueError(error_msg)
        if not args.pdf_path.is_dir():
            error_msg = f"PDF path must be a directory, not a file: {args.pdf_path}"
            raise ValueError(error_msg)
        pdf_path = args.pdf_path

    # Validate markdown_path if provided
    if args.markdown_path:
        if not args.markdown_path.exists():
            error_msg = f"Markdown directory does not exist: {args.markdown_path}"
            raise ValueError(error_msg)
        if not args.markdown_path.is_dir():
            error_msg = (
                f"Markdown path must be a directory, not a file: {args.markdown_path}"
            )
            raise ValueError(error_msg)
        markdown_path = args.markdown_path

    # Auto-generate output path if not provided
    if not args.output_path:
        args.output_path = eppi_json_path.parent
        logger.info(f"Auto-generated output path: {args.output_path}")
    output_path = args.output_path

    if markdown_path is None and pdf_path is not None:
        markdown_path = pdf_path / "markdown"
        markdown_path.mkdir(parents=True, exist_ok=True)
    # elif markdown_path is not None and pdf_path is None:
    #     pdf_path = None

    parse_pdf_stage = stage_from_job(
        jobify(
            name="parse_pdf",
            func_kwargs={
                "pdf_path": pdf_path,
                "out_path": markdown_path,
            },
        )(parse_pdf)
    )
    ingest_gs_stage = stage_from_job(
        jobify(
            name="ingest_gs",
            func_kwargs={
                "eppi_json_path": eppi_json_path,
                "output_dir": output_path,
                "csv_path": csv_path,
            },
        )(ingest_gold_standard_func)
    )
    llm_extraction_stage = stage_from_job(
        jobify(
            name="llm_extraction",
            job_type=JobType.EXTRACTION,
            func_kwargs={
                "markdown_dir": markdown_path,
                "attributes_file_path": output_path
                / DEFAULT_BASE_OUTPUT_DIR
                / DEFAULT_ATTRIBUTES_FILENAME,
                "output_path": output_path / "llm_extractions.json",
                "pdf_dir": pdf_path,
                "prompt_outfile": output_path / "full_prompt_payload.json",
            },
        )(llm_data_extraction)
    )

    pipeline = Pipeline(
        name="csv_02_batch_extraction",
        stages=[parse_pdf_stage, ingest_gs_stage, llm_extraction_stage],
    )
    pipeline.run()


if __name__ == "__main__":
    main()
