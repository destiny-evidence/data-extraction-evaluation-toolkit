"""Helper functions to run extraction via the CLI."""

import datetime
from collections.abc import Sequence
from pathlib import Path

import typer
import yaml
from loguru import logger
from pydantic import ValidationError

from deet.data_models.documents import ContextType, Document
from deet.data_models.project import ExperimentArtefacts
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy
from deet.ui import fail_with_message, notify


def load_config_from_context(
    ctx: typer.Context, config_path: Path | None
) -> DataExtractionConfig:
    """Load config from project context or path, or fail informatively."""
    if config_path is None:
        if not ctx.obj.project:
            no_config = (
                "This command is being run outside of a deet project, "
                "and no config file has been provided. Either run this "
                "from a project directory, or provide a config file."
            )
            fail_with_message(no_config)
        config_path = ctx.obj.project.config_path

    try:
        return DataExtractionConfig.from_yaml(config_path)
    except FileNotFoundError:
        fail_with_message(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        fail_with_message(f"YAML Syntax Error in {config_path}:\n{e}")
    except ValidationError as e:
        fail_with_message(f"Config validation error in {config_path}:\n{e}")


def init_extraction_run(
    out_dir: Path,
    run_name: str,
) -> ExperimentArtefacts:
    """Set up ID, folder and logging for data extraction run."""
    extraction_run_id = (
        datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        + f"_{run_name}"
    )

    experiment_out_dir = out_dir / extraction_run_id
    experiment_out_dir.mkdir(parents=True)

    logger.add(experiment_out_dir / "deet.log", level="DEBUG")

    return ExperimentArtefacts(base_dir=experiment_out_dir, run_id=extraction_run_id)


def prepare_documents(
    documents: Sequence[Document],
    config: DataExtractionConfig,
    linked_document_path: Path,
    pdf_dir: Path | None,
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
            notify(f"Linking documents using link map: {link_map_path}")
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
            notify(f"Loading linked documents from {linked_document_path}")
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
