"""Helper functions to run extraction via the CLI."""

import datetime
from collections.abc import Sequence
from pathlib import Path

import typer
import yaml
from loguru import logger
from pydantic import ValidationError

from deet.data_models.documents import ContextType, Document
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.extractors.llm_data_extractor import (
    DataExtractionConfig,
    ExtractionRunOutput,
    LLMDataExtractor,
)
from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy
from deet.ui import fail_with_message, notify
from deet.ui.terminal import console, render_template
from deet.ui.terminal.components import info_panel
from deet.ui.terminal.wizards import continue_after_key, run_model_wizard


def load_config_from_typer_context(
    typer_context: typer.Context, config_path: Path | None
) -> DataExtractionConfig:
    """Load config from project context or path, or fail informatively."""
    if config_path is None:
        if not typer_context.obj.project:
            no_config = (
                "This command is being run outside of a deet project, "
                "and no config file has been provided. Either run this "
                "from a project directory, or provide a config file."
            )
            fail_with_message(no_config)
        console.clear()
        console.print(
            info_panel(
                render_template("extraction/config_init"),
                "Data extraction config wizard",
            )
        )
        continue_after_key()
        return run_model_wizard(DataExtractionConfig)
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
        if linked_document_path.exists():
            notify(f"Loading linked documents from {linked_document_path}")
            documents = [Document.load(f) for f in linked_document_path.glob("*.json")]
            if documents:
                return documents

            notify(f"Couldn't find linked documents in {linked_document_path}")
        if pdf_dir is None:
            no_linked_docs_no_pdf = (
                "Full text extraction specified but"
                " linked document path does not contain documents,"
                " and no pdf dir supplied"
            )
            fail_with_message(no_linked_docs_no_pdf)

        if link_map_path is None:
            fail_with_message(
                "No link map supplied"
                f" and no linked documents in {linked_document_path}"
            )
        else:
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

    else:
        message = f"context type {config.default_context_type} not supported"
        fail_with_message(message)

    return None


def run_extraction_pipeline(
    typer_context: typer.Context,
    config_path: Path | None = None,
    prompt_population: CustomPromptPopulationMethod
    | None = CustomPromptPopulationMethod.FILE,
    run_name: str = "",
) -> tuple[ExtractionRunOutput, ProcessedAnnotationData, ExperimentArtefacts]:
    """Run the standard data extraction pipeline from the CLI."""
    import yaml

    from deet.extractors.cli_helpers import (
        init_extraction_run,
        load_config_from_typer_context,
        prepare_documents,
    )

    deet_project: DeetProject = typer_context.obj.project
    processed_annotation_data = deet_project.process_data()

    config = load_config_from_typer_context(typer_context, config_path)

    experiment_artefacts = init_extraction_run(deet_project.experiments_dir, run_name)

    if prompt_population is not None:
        processed_annotation_data.populate_custom_prompts(
            method=prompt_population, filepath=deet_project.prompt_csv_path
        )
        if not processed_annotation_data.attributes:
            fail_with_message(
                "No attributes selected. Perhaps you forgot to edit your prompt file"
            )

    data_extractor = LLMDataExtractor(config=config)

    documents = prepare_documents(
        processed_annotation_data.documents,
        config,
        linked_document_path=deet_project.linked_documents_path,
        pdf_dir=deet_project.pdf_dir,
        link_map_path=deet_project.link_map_path,
    )

    run_output = data_extractor.extract_from_documents(
        attributes=processed_annotation_data.attributes,
        documents=documents,
        context_type=data_extractor.config.default_context_type,
        output_file=experiment_artefacts.llm_annotations,
        show_progress=True,
    )

    processed_annotation_data.export_attributes_csv_file(
        experiment_artefacts.prompts_snapshot
    )

    experiment_artefacts.config_snapshot.write_text(
        yaml.safe_dump(data_extractor.config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    return run_output, processed_annotation_data, experiment_artefacts
