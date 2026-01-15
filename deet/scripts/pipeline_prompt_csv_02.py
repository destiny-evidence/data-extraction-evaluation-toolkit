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

from deet.data_models.base import Attribute, GoldStandardAnnotation

# @sagaruprety note that we now only use Eppi types in our
# specific use-case (i.e. a pipeline script), no longer in the
# underlying application. The application uses base.py data types.
from deet.data_models.eppi import EppiAttribute
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
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


def ingest_gold_standard_import_csv_func(
    eppi_json_path: Path, output_dir: Path, csv_path: Path
) -> None:
    """Convert EPPI JSON to DEET data models."""
    out = converter.process_annotation_file(eppi_json_path)

    # import prompts from csv for attribute population.
    if csv_path.parent == Path("."):  # noqa: PTH201
        csv_path = output_dir / csv_path

    out.populate_custom_prompts(method="file", filepath=csv_path)
    converter.save_processed_data(processed_data=out, output_dir=output_dir)


def llm_data_extraction(
    full_text_path: Path,
    attributes_file_path: Path,
    output_path: Path,
    file_path: Path | None = None,
    filter_by_attribute_ids: list[int] | None = None,
    **kwargs,
) -> dict[str, list[GoldStandardAnnotation]]:
    """
    Run LLM data extraction.

    Args:
        full_text_path: Path to markdown file with full text
        attributes_file_path: Path to attributes JSON file
        output_path: Path to save output JSON
        file_path: Path to the source file being processed (used as key in output)
        filter_by_attribute_ids: Optional list of attribute IDs to filter
        **kwargs: Additional arguments passed to extractor

    Returns:
        Dictionary mapping file paths to lists of annotations

    """
    full_text = full_text_path.read_text(encoding="utf-8")

    attributes_raw = json.loads(attributes_file_path.read_text(encoding="utf-8"))

    attributes: list[Attribute] = [EppiAttribute(**record) for record in attributes_raw]
    if filter_by_attribute_ids:
        attributes = [
            a for a in attributes if a.attribute_id in filter_by_attribute_ids
        ]

    # Use file_path if provided, otherwise derive from full_text_path
    if file_path is None:
        file_path = full_text_path

    return data_extractor.extract_from_documents(
        attributes=attributes,
        output_file=output_path,
        file_path=file_path,
        full_text=full_text,
        **kwargs,
    )


def process_directory(  # noqa: PLR0912, PLR0913
    eppi_json_path: Path,
    csv_path: Path,
    output_path: Path,
    pdf_dir: Path | None = None,
    markdown_dir: Path | None = None,
    filter_by_attribute_ids: list[int] | None = None,
    **kwargs,
) -> dict[str, list[GoldStandardAnnotation]]:
    """
    Process all PDFs or markdowns in a directory.

    Args:
        eppi_json_path: Path to EPPI JSON file
        csv_path: Path to CSV file with prompts
        output_path: Path to save combined output JSON
        pdf_dir: Directory containing PDF files (optional)
        markdown_dir: Directory containing markdown files
            (optional, for checking existing markdowns)
        filter_by_attribute_ids: Optional list of attribute IDs to filter
        **kwargs: Additional arguments passed to extraction

    Returns:
        Dictionary mapping file paths to lists of annotations

    """
    eppi_out_path = output_path.parent / "eppi"

    # Ingest gold standard with CSV import once
    ingest_gold_standard_import_csv_func(eppi_json_path, eppi_out_path, csv_path)

    all_results: dict[str, list[GoldStandardAnnotation]] = {}

    # Determine which directory to process
    if pdf_dir and pdf_dir.exists() and pdf_dir.is_dir():
        input_dir = pdf_dir
        file_extensions = [".pdf"]
        is_pdf = True
    elif markdown_dir and markdown_dir.exists() and markdown_dir.is_dir():
        input_dir = markdown_dir
        file_extensions = [".md"]
        is_pdf = False
    else:
        error_msg = "Either pdf_dir or markdown_dir must be provided and exist"
        raise ValueError(error_msg)

    # Find all files with matching extensions
    input_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in file_extensions
    ]

    if not input_files:
        logger.warning(f"No {file_extensions} files found in {input_dir}")
        return all_results

    logger.info(f"Processing {len(input_files)} files from {input_dir}")
    logger.info(f"Files to process: {[f.name for f in sorted(input_files)]}")

    for input_file in sorted(input_files):
        logger.info(f"Processing file: {input_file.name} ({input_file})")
        try:
            # Determine markdown path
            if is_pdf:
                # Check if markdown exists in markdown_dir
                md_filename = input_file.stem + ".md"
                if markdown_dir and (markdown_dir / md_filename).exists():
                    md_path = markdown_dir / md_filename
                    logger.info(f"Using existing markdown: {md_path}")
                else:
                    # Parse PDF to markdown
                    # Save to markdown_dir if provided, otherwise to PDF's directory
                    if markdown_dir:
                        md_path = markdown_dir / md_filename
                        # Ensure markdown_dir exists
                        markdown_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        md_path = input_file.parent / md_filename
                    logger.info(f"Parsing PDF to markdown: {md_path}")
                    parse_pdf(input_file, md_path, skip_if_md_exists=False)
            else:
                md_path = input_file

            # Run extraction for this file
            file_output_path = eppi_out_path / f"{input_file.stem}_llm_extractions.json"
            file_results = llm_data_extraction(
                full_text_path=md_path,
                attributes_file_path=eppi_out_path / "attributes.json",
                output_path=file_output_path,
                file_path=input_file,
                filter_by_attribute_ids=filter_by_attribute_ids,
                prompt_outfile=eppi_out_path
                / f"{input_file.stem}_full_prompt_payload.json",
                **kwargs,
            )
            all_results.update(file_results)
            logger.info(
                f"Successfully processed {input_file.name}: "
                f"{len(file_results.get(str(input_file.resolve()), []))} annotations"
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to process {input_file}: {e}")
            logger.debug("Error details", exc_info=True)
            continue

    logger.info(
        f"Completed processing. Total files processed: {len(all_results)} "
        f"out of {len(input_files)}"
    )

    # Save combined results
    if all_results:
        # Convert annotations to dict format for JSON serialization
        results_json = {
            file_path: [ann.model_dump() for ann in annotations]
            for file_path, annotations in all_results.items()
        }
        output_path.write_text(json.dumps(results_json, indent=2))
        logger.info(f"Combined results saved to: {output_path}")

    return all_results


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
        help="path you want to write prompt editing csv to.",
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

    # Auto-generate output path if not provided
    if not args.output_path:
        eppi_json_dir = str(Path(args.eppi_json_path).name).split(".")[:-1][0]
        input_dir = args.pdf_path or args.markdown_path
        if input_dir:
            args.output_path = (
                input_dir.parent
                / "tmp_parsed_eppi"
                / eppi_json_dir
                / "llm_extractions.json"
            )
        else:
            args.output_path = Path("llm_extractions.json")
        logger.info(f"Auto-generated output path: {args.output_path}")

    process_directory(
        eppi_json_path=args.eppi_json_path,
        csv_path=args.csv_path,
        output_path=args.output_path,
        pdf_dir=args.pdf_path,
        markdown_dir=args.markdown_path,
    )


if __name__ == "__main__":
    main()
