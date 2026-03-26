"""Helper functions to run extraction via the CLI."""

import datetime
from collections.abc import Sequence
from pathlib import Path

import yaml  # type:ignore[import-untyped]
from loguru import logger

from deet.data_models.documents import ContextType, Document
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy
from deet.utils.cli_utils import echo_and_log, fail_with_message


def load_or_init_config(config_path: Path) -> DataExtractionConfig:
    """Load config, or initialise default config."""
    if config_path.exists():
        config = DataExtractionConfig(**yaml.safe_load(config_path.read_text()))
    else:
        echo_and_log(
            f"Config file: {config_path} does not exist."
            " Initialising config with default settings."
        )
        config = DataExtractionConfig()
    return config


def init_extraction_run(
    out_dir: Path,
    run_name: str,
) -> tuple[str, Path]:
    """Set up ID, folder and logging for data extraction run."""
    extraction_run_id = (
        datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        + f"_{run_name}"
    )

    experiment_out_dir = out_dir / extraction_run_id
    experiment_out_dir.mkdir(parents=True)

    logger.add(experiment_out_dir / "deet.log", level="DEBUG")

    return extraction_run_id, experiment_out_dir


def prepare_documents(
    documents: Sequence[Document],
    config: DataExtractionConfig,
    linked_document_path: Path,
    pdf_dir: Path,
    link_map_path: Path | None,
) -> Sequence[Document]:
    """
    Load documents depending on the context type we want.

    NOTE: while there are no arg-defaults defined here,
    when used in cli.py, we populate defaults via
    typer arg defaults.

    If fulltext, try to load linked documents, or create them if not.
    """
    if config.default_context_type == ContextType.ABSTRACT_ONLY:
        return documents
    if config.default_context_type == ContextType.FULL_DOCUMENT:
        if link_map_path is not None:
            echo_and_log(f"Linking documents using link map: {link_map_path}")
            linker = DocumentReferenceLinker(
                references=documents,
                document_base_dir=pdf_dir,
                document_reference_mapping=link_map_path,
                linking_strategies=[LinkingStrategy.MAPPING_FILE],
            )
            documents = linker.link_many_references_parsed_documents()
            for linked_document in documents:
                file_path = (
                    linked_document_path
                    / f"{linked_document.safe_identity.document_id}.json"
                )
                linked_document.save(file_path)

            if not documents:
                no_links = (
                    f"context type {config.default_context_type} selected"
                    " but no linked documents could be found or created"
                )
                fail_with_message(no_links)

            return documents
        if linked_document_path.exists():
            echo_and_log(f"Loading linked documents from {linked_document_path}")
            documents = [Document.load(f) for f in linked_document_path.glob("*.json")]
            if documents:
                return documents
            fail_with_message(
                "Linked document path does not contain any linked documents."
                " Please use a link map"
                " to create linked documents "
                "(`deet init-linkage-mapping-file`)"
            )
        else:
            fail_with_message(
                "Linked document path does not exist. Please use a link map"
                " to create linked documents "
                "(`deet init-linkage-mapping-file`)"
            )
    else:
        message = f"context type {config.default_context_type} not supported"
        fail_with_message(message)

    return None
