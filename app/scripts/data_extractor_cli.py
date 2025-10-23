#!/usr/bin/env python3
"""CLI for running the DataExtractionModule with flexible parameters."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from app.extractors.data_extraction_module import (
    AttributeSelectionMode,
    ContextType,
    DataExtractionConfig,
    DataExtractionModule,
)
from app.logger import logger
from app.models.eppi import EppiAttribute, EppiDocument, EppiGoldStandardAnnotation

# Load environment variables
load_dotenv(override=True)


class SimpleConfig:
    """Simple configuration class for YAML config files."""

    def __init__(self, config_data: dict[str, Any]) -> None:
        """Initialize from config data dictionary."""
        self.model = config_data.get("model", "gpt-4o-mini")
        self.temperature = config_data.get("temperature", 0.1)
        self.max_tokens = config_data.get("max_tokens")
        self.max_context_length = config_data.get("max_context_length", 40000)
        self.context_type = config_data.get("context_type", "full_document")
        self.include_reasoning = config_data.get("include_reasoning", True)
        self.include_additional_text = config_data.get("include_additional_text", True)

        # File paths
        self.annotations_file = config_data.get("annotations_file")
        self.attributes_file = config_data.get("attributes_file")
        self.documents_file = config_data.get("documents_file")
        self.system_prompt_file = config_data.get("system_prompt_file")
        self.output_file = config_data.get("output_file", "extraction_results.json")

        # Data filtering
        self.attribute_ids_file = config_data.get("attribute_ids_file")
        self.document_ids_file = config_data.get("document_ids_file")

        # Logging
        self.verbose = config_data.get("verbose", False)
        self.log_file = config_data.get("log_file")

    def load_attribute_ids(self) -> list[str]:
        """Load attribute IDs from file or return empty list for all."""
        if not self.attribute_ids_file:
            return []

        try:
            with Path(self.attribute_ids_file).open("r") as f:
                ids = [line.strip() for line in f if line.strip()]
            logger.info(
                f"Loaded {len(ids)} attribute IDs from {self.attribute_ids_file}"
            )
        except (OSError, FileNotFoundError, ValueError) as e:
            logger.error(
                f"Error loading attribute IDs from {self.attribute_ids_file}: {e}"
            )
            return []
        else:
            return ids

    def load_document_ids(self) -> list[str]:
        """Load document IDs from file or return empty list for all."""
        if not self.document_ids_file:
            return []

        try:
            with Path(self.document_ids_file).open("r") as f:
                ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(ids)} document IDs from {self.document_ids_file}")
        except (OSError, FileNotFoundError, ValueError) as e:
            logger.error(
                f"Error loading document IDs from {self.document_ids_file}: {e}"
            )
            return []
        else:
            return ids

    @classmethod
    def from_yaml(cls, config_path: Path) -> "SimpleConfig":
        """Load configuration from YAML file."""
        if not config_path.exists():
            error_msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading configuration from: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls(config_data)

    @classmethod
    def create_template(cls, output_path: Path) -> None:
        """Create a configuration template file."""
        template_data = {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": None,
            "max_context_length": 40000,
            "context_type": "full_document",
            "include_reasoning": True,
            "include_additional_text": True,
            "annotations_file": "data/annotated_documents.json",
            "attributes_file": "data/attributes.json",
            "documents_file": "data/documents.json",
            "system_prompt_file": None,
            "output_file": "extraction_results.json",
            "attribute_ids_file": None,
            "document_ids_file": None,
            "verbose": False,
            "log_file": None,
        }

        template_content = f"""# Data Extraction Configuration
# Simple configuration for running data extraction experiments

# Model Configuration
model: "{template_data['model']}"  # LLM model to use (e.g., gpt-4o-mini, gpt-4, claude-3)
temperature: {template_data['temperature']}  # Temperature for LLM generation (0.0-2.0)
max_tokens: {template_data['max_tokens']}  # Maximum tokens for LLM response (null for no limit)
max_context_length: {template_data['max_context_length']}  # Maximum context length

# Extraction Configuration
context_type: "{template_data['context_type']}"  # full_document, abstract, custom
include_reasoning: {str(template_data['include_reasoning']).lower()}  # Include reasoning in output
include_additional_text: {str(template_data['include_additional_text']).lower()}  # Include additional text in output

# File Paths
annotations_file: "{template_data['annotations_file']}"  # Path to annotated_documents.json
attributes_file: "{template_data['attributes_file']}"  # Path to attributes.json
documents_file: "{template_data['documents_file']}"  # Path to documents.json
system_prompt_file: {template_data['system_prompt_file']}  # Path to custom system prompt file (null for default)
output_file: "{template_data['output_file']}"  # Path to save results

# Data Filtering (null = process all)
attribute_ids_file: {template_data['attribute_ids_file']}  # Path to file with attribute IDs (one per line)
document_ids_file: {template_data['document_ids_file']}  # Path to file with document IDs (one per line)

# Logging Configuration
verbose: {str(template_data['verbose']).lower()}  # Enable verbose logging
log_file: {template_data['log_file']}  # Path to log file (null for console only)
"""

        with output_path.open("w", encoding="utf-8") as f:
            f.write(template_content)

        logger.info(f"Configuration template created at: {output_path}")


def load_processed_data(
    annotations_file: Path,
    attributes_file: Path,
    documents_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load processed EPPI data from JSON files.

    Args:
        annotations_file: Path to annotated_documents.json
        attributes_file: Path to attributes.json
        documents_file: Path to documents.json

    Returns:
        Tuple of (annotations_data, attributes_data, documents_data)

    """
    logger.info("Loading data from:")
    logger.info(f"  - Annotations: {annotations_file}")
    logger.info(f"  - Attributes: {attributes_file}")
    logger.info(f"  - Documents: {documents_file}")

    with annotations_file.open() as f:
        annotations_data = json.load(f)

    with attributes_file.open() as f:
        attributes_data = json.load(f)

    with documents_file.open() as f:
        documents_data = json.load(f)

    logger.info(
        f"Loaded {len(annotations_data)} annotated documents, {len(attributes_data)} attributes, {len(documents_data)} documents"
    )

    return annotations_data, attributes_data, documents_data


def convert_to_models(
    attributes_data: list[dict[str, Any]],
    documents_data: list[dict[str, Any]],
    selected_attribute_ids: list[str] | None = None,
    selected_document_ids: list[str] | None = None,
) -> tuple[list[EppiAttribute], list[EppiDocument]]:
    """
    Convert JSON data to Pydantic models with optional filtering.

    Args:
        attributes_data: Raw attributes data
        documents_data: Raw documents data
        selected_attribute_ids: Optional list of attribute IDs to filter
        selected_document_ids: Optional list of document IDs to filter

    Returns:
        Tuple of (attributes, documents)

    """
    # Convert attributes
    attributes = []
    for attr_data in attributes_data:
        if (
            selected_attribute_ids
            and attr_data.get("attribute_id") not in selected_attribute_ids
        ):
            continue

        processed_attr_data = {
            "question_target": "",
            "output_data_type": "bool",
            "attribute_id": str(attr_data.get("attribute_id", "")),
            "attribute_label": attr_data.get("attribute_label", ""),
            "attribute_set_description": attr_data.get("attribute_set_description"),
            "attribute_type": attr_data.get("attribute_type"),
            "hierarchy_path": attr_data.get("hierarchy_path"),
            "hierarchy_level": attr_data.get("hierarchy_level", 0),
            "is_leaf": attr_data.get("is_leaf", True),
            "parent_attribute_id": attr_data.get("parent_attribute_id"),
        }
        attribute = EppiAttribute.model_validate(processed_attr_data)
        attributes.append(attribute)

    # Convert documents
    documents = []
    for doc_data in documents_data:
        if (
            selected_document_ids
            and doc_data.get("document_id") not in selected_document_ids
        ):
            continue

        document = EppiDocument.model_validate(doc_data)
        documents.append(document)

    logger.info(
        f"Converted {len(attributes)} attributes and {len(documents)} documents"
    )

    return attributes, documents


def run_extraction(
    documents: list[EppiDocument],
    attributes: list[EppiAttribute],
    config: DataExtractionConfig,
    output_file: Path,
    batch_size: int = 1,
    custom_system_prompt_file: Path | None = None,
) -> list[EppiGoldStandardAnnotation]:
    """
    Run data extraction on documents and attributes.

    Args:
        documents: List of documents to process
        attributes: List of attributes to extract
        config: Data extraction configuration
        output_file: Path to save results
        batch_size: Number of documents to process in parallel

    Returns:
        List of extracted annotations

    """
    extractor = DataExtractionModule(config, custom_system_prompt_file)
    all_annotations = []

    logger.info(
        f"Starting extraction on {len(documents)} documents with {len(attributes)} attributes"
    )
    logger.info(f"Batch size: {batch_size}")

    for i, document in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}: {document.name}")

        try:
            # Extract data for this document
            annotations = extractor.extract_from_document(document, attributes)
            all_annotations.extend(annotations)

            logger.info(f"Extracted {len(annotations)} annotations from document")

        except (ValueError, RuntimeError, KeyError) as e:
            logger.error(f"Error processing document {document.name}: {e}")
            continue

    # Save results
    logger.info(f"Saving {len(all_annotations)} annotations to {output_file}")

    results_data = {
        "config": config.model_dump(),
        "total_documents": len(documents),
        "total_attributes": len(attributes),
        "total_annotations": len(all_annotations),
        "annotations": [ann.model_dump() for ann in all_annotations],
    }

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"Results saved to: {output_file}")

    return all_annotations


def main() -> None:
    """Run the main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run data extraction using the DataExtractionModule",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config (easiest - just run!)
  python data_extractor_cli.py

  # Using custom YAML config file
  python data_extractor_cli.py --config my_config.yaml

  # Create a config template
  python data_extractor_cli.py --create-template my_config.yaml

  # Using command-line arguments (legacy)
  python data_extractor_cli.py --annotations-file data/annotated_documents.json --attributes-file data/attributes.json --documents-file data/documents.json

  # Extract specific attributes from specific documents
  python data_extractor_cli.py --annotations-file data/annotated_documents.json --attributes-file data/attributes.json --documents-file data/documents.json --attribute-ids 123,456 --document-ids doc1,doc2
        """,
    )

    # Config file support
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file (alternative to command-line args)",
    )
    parser.add_argument(
        "--create-template", type=Path, help="Create a configuration template file"
    )

    # Required arguments (only required if not using config file)
    parser.add_argument(
        "--annotations-file", type=Path, help="Path to annotated_documents.json file"
    )
    parser.add_argument(
        "--attributes-file", type=Path, help="Path to attributes.json file"
    )
    parser.add_argument(
        "--documents-file", type=Path, help="Path to documents.json file"
    )

    # Optional filtering arguments
    parser.add_argument(
        "--attribute-ids",
        type=str,
        help="Comma-separated list of attribute IDs to extract (default: all)",
    )
    parser.add_argument(
        "--document-ids",
        type=str,
        help="Comma-separated list of document IDs to process (default: all)",
    )

    # Configuration arguments
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Path to custom system prompt file (default: app/prompts/system_prompt_v0.txt)",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "single", "batch", "by_names", "by_ids"],
        default="all",
        help="Attribute selection mode (default: all)",
    )
    parser.add_argument(
        "--context-type",
        choices=["full_document", "abstract", "custom"],
        default="full_document",
        help="Context type to use (default: full_document)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM generation (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens for LLM response"
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=40000,
        help="Maximum context length (default: 40000)",
    )

    # Output arguments
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save results (default: extraction_results.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of documents to process in parallel (default: 1)",
    )

    # Control arguments
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=True,
        help="Include reasoning in output (default: True)",
    )
    parser.add_argument(
        "--include-additional-text",
        action="store_true",
        default=True,
        help="Include additional text in output (default: True)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Handle template creation
    if args.create_template:
        SimpleConfig.create_template(args.create_template)
        logger.info(f"Configuration template created at: {args.create_template}")
        return

    # Handle config file mode (or use default if no args provided)
    if args.config or (
        not args.annotations_file
        and not args.attributes_file
        and not args.documents_file
    ):
        config_file = args.config or Path("default_config.yaml")
        try:
            config = SimpleConfig.from_yaml(config_file)

            # Set up logging
            if config.verbose:
                import logging

                logging.getLogger().setLevel(logging.DEBUG)

            # Set up logging file if specified
            if config.log_file:
                import logging

                file_handler = logging.FileHandler(config.log_file)
                file_handler.setLevel(logging.INFO)
                logging.getLogger().addHandler(file_handler)

            # Load data
            if not all(
                [config.annotations_file, config.attributes_file, config.documents_file]
            ):
                raise ValueError("Missing required file paths in configuration")

            # Type assertions for mypy
            assert config.annotations_file is not None
            assert config.attributes_file is not None
            assert config.documents_file is not None

            annotations_data, attributes_data, documents_data = load_processed_data(
                Path(config.annotations_file),
                Path(config.attributes_file),
                Path(config.documents_file),
            )

            # Load filtering IDs
            selected_attribute_ids = config.load_attribute_ids()
            selected_document_ids = config.load_document_ids()

            # Convert to models with filtering
            attributes, documents = convert_to_models(
                attributes_data,
                documents_data,
                selected_attribute_ids if selected_attribute_ids else None,
                selected_document_ids if selected_document_ids else None,
            )

            if not attributes:
                logger.error("No attributes found matching the criteria")
                sys.exit(1)

            if not documents:
                logger.error("No documents found matching the criteria")
                sys.exit(1)

            # Create DataExtractionConfig for the module
            module_config = DataExtractionConfig(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                context_type=ContextType(config.context_type),
                max_context_length=config.max_context_length,
                attribute_selection_mode=AttributeSelectionMode.BATCH,  # Always batch mode
                selected_attribute_ids=selected_attribute_ids,
                include_reasoning=config.include_reasoning,
                include_additional_text=config.include_additional_text,
            )

            # Create extractor with custom system prompt if provided
            custom_system_prompt_file = (
                Path(config.system_prompt_file) if config.system_prompt_file else None
            )
            extractor = DataExtractionModule(module_config, custom_system_prompt_file)

            # Run extraction
            logger.info("=" * 60)
            logger.info("STARTING DATA EXTRACTION")
            logger.info("=" * 60)

            all_annotations = []
            for i, document in enumerate(documents):
                logger.info(
                    f"Processing document {i+1}/{len(documents)}: {document.name}"
                )

                try:
                    annotations = extractor.extract_from_document(document, attributes)
                    all_annotations.extend(annotations)
                    logger.info(
                        f"Extracted {len(annotations)} annotations from document"
                    )
                except (ValueError, RuntimeError, KeyError) as e:
                    logger.error(f"Error processing document {document.name}: {e}")
                    continue

            # Save results
            logger.info(
                f"Saving {len(all_annotations)} annotations to {config.output_file}"
            )

            results_data = {
                "config": {
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "max_context_length": config.max_context_length,
                    "context_type": config.context_type,
                    "include_reasoning": config.include_reasoning,
                    "include_additional_text": config.include_additional_text,
                    "annotations_file": str(Path(config.annotations_file).absolute()),
                    "attributes_file": str(Path(config.attributes_file).absolute()),
                    "documents_file": str(Path(config.documents_file).absolute()),
                    "system_prompt_file": str(
                        Path(config.system_prompt_file).absolute()
                    )
                    if config.system_prompt_file
                    else None,
                    "output_file": config.output_file,
                    "attribute_ids_file": str(
                        Path(config.attribute_ids_file).absolute()
                    )
                    if config.attribute_ids_file
                    else None,
                    "document_ids_file": str(Path(config.document_ids_file).absolute())
                    if config.document_ids_file
                    else None,
                    "verbose": config.verbose,
                    "log_file": config.log_file,
                },
                "total_documents": len(documents),
                "total_attributes": len(attributes),
                "total_annotations": len(all_annotations),
                "annotations": [ann.model_dump() for ann in all_annotations],
            }

            assert config.output_file is not None
            output_path = Path(config.output_file)
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open("w") as f:
                json.dump(results_data, f, indent=2)

            logger.info("=" * 60)
            logger.info("EXTRACTION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total annotations extracted: {len(all_annotations)}")
            logger.info(f"Results saved to: {config.output_file}")

        except (ValueError, RuntimeError, KeyError, FileNotFoundError) as e:
            logger.error(f"Error during extraction: {e}")
            sys.exit(1)

        return

    # Legacy command-line argument mode
    # Validate required arguments
    if not args.annotations_file or not args.attributes_file or not args.documents_file:
        parser.error(
            "--annotations-file, --attributes-file, and --documents-file are required when not using --config"
        )

    # Set up logging
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    # Validate file paths
    for file_path in [args.annotations_file, args.attributes_file, args.documents_file]:
        if file_path and not file_path.exists():
            logger.error(f"File not found: {file_path}")
            sys.exit(1)

    # Parse attribute and document IDs
    selected_attribute_ids: list[str] | None = None
    if args.attribute_ids:
        selected_attribute_ids = [
            attr_id.strip() for attr_id in args.attribute_ids.split(",")
        ]
        logger.info(f"Selected attribute IDs: {selected_attribute_ids}")

    selected_document_ids: list[str] | None = None
    if args.document_ids:
        selected_document_ids = [
            doc_id.strip() for doc_id in args.document_ids.split(",")
        ]
        logger.info(f"Selected document IDs: {selected_document_ids}")

    # Set up output file
    if not args.output_file:
        args.output_file = Path("extraction_results.json")

    try:
        # Load data
        annotations_data, attributes_data, documents_data = load_processed_data(
            args.annotations_file, args.attributes_file, args.documents_file
        )

        # Convert to models with filtering
        attributes, documents = convert_to_models(
            attributes_data,
            documents_data,
            selected_attribute_ids,
            selected_document_ids,
        )

        if not attributes:
            logger.error("No attributes found matching the criteria")
            sys.exit(1)

        if not documents:
            logger.error("No documents found matching the criteria")
            sys.exit(1)

        # Set up configuration
        config = DataExtractionConfig(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            context_type=ContextType(args.context_type),
            max_context_length=args.max_context_length,
            attribute_selection_mode=AttributeSelectionMode(args.mode),
            selected_attribute_ids=selected_attribute_ids or [],
            include_reasoning=args.include_reasoning,
            include_additional_text=args.include_additional_text,
        )

        # Override system prompt file if specified
        custom_system_prompt_file = None
        if args.system_prompt_file:
            if not args.system_prompt_file.exists():
                logger.error(f"System prompt file not found: {args.system_prompt_file}")
                sys.exit(1)
            custom_system_prompt_file = args.system_prompt_file
            logger.info(f"Using custom system prompt: {args.system_prompt_file}")

        # Run extraction
        logger.info("=" * 60)
        logger.info("STARTING DATA EXTRACTION")
        logger.info("=" * 60)

        annotations = run_extraction(
            documents,
            attributes,
            config,
            args.output_file,
            args.batch_size,
            custom_system_prompt_file,
        )

        logger.info("=" * 60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total annotations extracted: {len(annotations)}")
        logger.info(f"Results saved to: {args.output_file}")

    except (ValueError, RuntimeError, KeyError, FileNotFoundError) as e:
        logger.error(f"Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
