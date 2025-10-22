#!/usr/bin/env python3
"""
Simple CLI script using code in processors/eppi_annotation_converter.py
to convert EPPI-json files into DEET's native format.
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from app.processors.eppi_annotation_converter import EppiAnnotationConverter


def main() -> None:
    """Run the annotation converter CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert EPPI annotations to structured format. "
            "Creates organized directory structure: output_dir/eppi/{filename}/"
        ),
        epilog=(
            "Example: python annotation_converter_cli.py input.json output/ "
            "creates output/eppi/input/ with JSON files"
        ),
    )

    parser.add_argument(
        "-i", "--input_file", type=str, help="Path to the raw EPPI JSON file"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",  # Explicitly name the destination variable
        help=(
            "Base directory to save processed files "
            "(creates `output_dir/eppi/{filename}/` structure)"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    converter = EppiAnnotationConverter()
    processed_data = converter.process_annotation_file(str(input_path))
    saved_files = converter.save_processed_data(
        processed_data, str(output_path), str(input_path)
    )

    logger.info("Conversion complete!")
    logger.info(f"Files saved to: {Path(saved_files['attributes']).parent.absolute()}")
    for file_type, file_path in saved_files.items():
        logger.info(f"  {file_type}: {Path(file_path).absolute()}")


if __name__ == "__main__":
    main()
