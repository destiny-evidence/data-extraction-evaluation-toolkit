#!/usr/bin/env python3
"""CLI script for converting EPPI annotation JSON files to structured format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.logger import logger
from app.processors.annotation_converter import AnnotationConverter


def main() -> None:
    """Run the annotation converter CLI."""
    parser = argparse.ArgumentParser(
        description="Convert EPPI annotations to structured format"
    )
    parser.add_argument("input_file", help="Path to the raw EPPI JSON file")
    parser.add_argument("output_dir", help="Directory to save processed files")
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process the annotation file
    converter = AnnotationConverter()
    processed_data = converter.process_annotation_file(str(input_path))
    saved_files = converter.save_processed_data(processed_data, str(output_path))

    logger.info("Conversion complete!")
    for file_type, file_path in saved_files.items():
        logger.info(f"  {file_type}: {file_path}")


if __name__ == "__main__":
    main()
