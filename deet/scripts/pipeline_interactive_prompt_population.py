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

from deet.data_models.base import Attribute, ContextType, GoldStandardAnnotation
from deet.data_models.eppi import EppiAttribute
from deet.data_models.pipeline import JobType, Pipeline, jobify, stage_from_job
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
from deet.processors.base_converter import (
    DEFAULT_ATTRIBUTES_FILENAME,
    DEFAULT_BASE_OUTPUT_DIR,
)
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter
from deet.processors.parser import DocumentParser

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
    Parse all PDFs in pdf_path (dir) to markdown files in out_path (dir).

    No-op if pdf_path or out_path is None (e.g. when only markdown dir is given).

    Args:
        pdf_path: Directory containing PDF files.
        out_path: Directory to write markdown files to.
        skip_if_md_exists: If True, skip parsing when markdown already exists.
            You are responsible for ensuring the md file matches the pdf.
            Defaults to True.

    """
    if pdf_path is None or out_path is None:
        logger.info(
            "pdf_path or out_path not provided; skipping parse_pdf stage (no-op)."
        )
        return
    if not pdf_path.is_dir():
        missing_paths = "must specify a pdf_path that is an existing directory"
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


def ingest_gold_standard_func(eppi_json_path: Path, output_dir: Path) -> None:
    """Convert EPPI JSON to DEET data models."""
    out = converter.process_annotation_file(eppi_json_path)
    converter.write_processed_data_to_file(processed_data=out, output_dir=output_dir)


def llm_data_extraction(
    markdown_dir: Path,
    attributes_file_path: Path,
    output_path: Path,
    filter_by_attribute_ids: list[int] | None = None,
    prompt_outfile: Path | None = None,
) -> dict[str, list[GoldStandardAnnotation]]:
    """
    Run LLM data extraction for all files in the markdown dir.

    Delegates to LLMDataExtractor.extract_from_documents with markdown_dir
    (and pdf_dir), which performs the loop over files inside the extractor.

    Args:
        markdown_dir: Directory of markdown files.
        attributes_file_path: Path to attributes JSON file.
        output_path: Path to save combined output JSON.
        pdf_dir: Directory of PDFs (optional); when set, lists inputs from here.
        filter_by_attribute_ids: Optional list of attribute IDs to filter.
        prompt_outfile: Optional file path to write final prompt sent to LLM.

    Returns:
        Dictionary mapping file paths to lists of annotations.

    """
    attributes_raw = json.loads(attributes_file_path.read_text(encoding="utf-8"))
    attributes: list[Attribute] = [EppiAttribute(**record) for record in attributes_raw]
    if filter_by_attribute_ids:
        attributes = [
            a for a in attributes if a.attribute_id in filter_by_attribute_ids
        ]

    for att in attributes:
        att.enter_custom_prompt()

    return data_extractor.extract_from_documents(
        attributes=attributes,
        markdown_dir=markdown_dir,
        output_file=output_path,
        context_type=ContextType.FULL_DOCUMENT,
        prompt_outfile=prompt_outfile,
    )


def main() -> None:
    """Run main. All inputs are directories (pdf dir and/or markdown dir)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pdf_path",
        help="directory containing PDF files",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-m",
        "--markdown_path",
        help="directory containing or for markdown files",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "-e", "--eppi_json_path", help="path to eppi json", type=Path, required=True
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="path to save output JSON (auto-generated if not provided)",
        type=Path,
        required=False,
    )

    args = parser.parse_args()

    if not args.pdf_path and not args.markdown_path:
        msg = "At least one of -p/--pdf_path or -m/--markdown_path must be provided"
        raise ValueError(msg)

    if args.pdf_path and (not args.pdf_path.exists() or not args.pdf_path.is_dir()):
        msg = f"PDF path must be an existing directory: {args.pdf_path}"
        raise ValueError(msg)
    if args.markdown_path and (
        not args.markdown_path.exists() or not args.markdown_path.is_dir()
    ):
        msg = f"Markdown path must be an existing directory: {args.markdown_path}"
        raise ValueError(msg)

    # Auto-generate output path if not provided
    if not args.output_path:
        eppi_json_dir = Path(args.eppi_json_path).stem
        input_dir = args.pdf_path or args.markdown_path
        if input_dir:
            args.output_path = input_dir.parent / "tmp_parsed_eppi" / eppi_json_dir
        else:
            args.output_path = Path("llm_extractions.json")
        logger.info(f"Auto-generated output path: {args.output_path}")

    if args.markdown_path is None and args.pdf_path is not None:
        args.markdown_path = args.output_path.parent / "markdown"
        args.markdown_path.mkdir(parents=True, exist_ok=True)

    logger.debug("decorating our functions as Jobs and PipelineStages")
    parse_pdf_stage = stage_from_job(
        jobify(
            name="parse_pdf",
            func_kwargs={
                "pdf_path": args.pdf_path,
                "out_path": args.markdown_path,
            },
        )(parse_pdf),
        skip_jobs_if_failed=False,
    )

    ingest_gs_stage = stage_from_job(
        jobify(
            name="ingest_gs",
            func_kwargs={
                "eppi_json_path": args.eppi_json_path,
                "output_dir": args.output_path,
            },
        )(ingest_gold_standard_func),
        skip_jobs_if_failed=False,
    )

    llm_extraction_stage = stage_from_job(
        jobify(
            name="llm_extraction",
            job_type=JobType.EXTRACTION,
            func_kwargs={
                "markdown_dir": args.markdown_path,
                "attributes_file_path": args.output_path
                / DEFAULT_BASE_OUTPUT_DIR
                / DEFAULT_ATTRIBUTES_FILENAME,
                "output_path": args.output_path / "llm_extractions.json",
                "prompt_outfile": args.output_path / "full_prompt_payload.json",
            },
        )(llm_data_extraction),
        skip_jobs_if_failed=False,
    )

    my_beautiful_pipeline = Pipeline(
        name="test_pipeline",
        stages=[parse_pdf_stage, ingest_gs_stage, llm_extraction_stage],
    )

    my_beautiful_pipeline.run()


if __name__ == "__main__":
    main()
