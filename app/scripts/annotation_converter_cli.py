#!/usr/bin/env python3
"""
Simple CLI script for the annotation converter.

This script imports and uses the existing AnnotationConverter class from the processor module.
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path so we can import app modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import app modules after path setup
from app.logger import logger  # noqa: E402
from app.processors.annotation_converter import AnnotationConverter  # noqa: E402


def main() -> None:
    """Run the annotation converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert EPPI annotations to structured format. Creates organized directory structure: output_dir/eppi/{filename}/",
        epilog="Example: python annotation_converter_cli.py input.json output/ creates output/eppi/input/ with JSON files",
    )
    parser.add_argument("input_file", help="Path to the raw EPPI JSON file")
    parser.add_argument(
        "output_dir",
        help="Base directory to save processed files (creates output_dir/eppi/{filename}/ structure)",
    )
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process the annotation file using the existing processor
    converter = AnnotationConverter()
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
