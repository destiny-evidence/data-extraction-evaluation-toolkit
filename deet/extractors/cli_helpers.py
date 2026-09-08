"""Helper functions to run extraction via the CLI."""

import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from deet.data_models.documents import ContextType, Document
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.extraction import (
    DocumentParsingStats,
    ExtractionPipelineStage,
    ExtractionRunOutput,
)
from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.extractor_registry import get_data_extractor
from deet.processors.directory_processor import create_documents_from_directory
from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy
from deet.ui import fail_with_message, notify
from deet.ui.terminal import console, render_template
from deet.ui.terminal.components import info_panel
from deet.ui.terminal.wizards import continue_after_key, run_model_wizard
from deet.utils.timing import measure_elapsed


def load_or_init_config(config_path: Path | None) -> DataExtractionConfig:
    """Load config from project context or path, or fail informatively."""
    if config_path is None:
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


def _cached_parsing_stats(
    documents: Sequence[Document],
) -> dict[str, DocumentParsingStats]:
    """Build skipped parsing stats for documents loaded from cache."""
    return {
        str(document.safe_identity.document_id): DocumentParsingStats()
        for document in documents
    }


def prepare_documents(
    documents: Sequence[Document],
    config: DataExtractionConfig,
    linked_document_path: Path,
    pdf_dir: Path | None,
    link_map_path: Path | None,
) -> tuple[Sequence[Document], dict[str, DocumentParsingStats]]:
    """
    Load documents depending on the context type we want.

    NOTE: while there are no arg-defaults defined here,
    when used in cli.py, we populate defaults via
    typer arg defaults.

    If fulltext, try to load linked documents, or create them if not.
    """
    if config.default_context_type == ContextType.ABSTRACT_ONLY:
        return documents, {}
    if config.default_context_type == ContextType.FULL_DOCUMENT:
        if linked_document_path.exists():
            notify(f"Loading linked documents from {linked_document_path}")
            linked_documents = []
            for document in documents:
                document_id = document.safe_identity.document_id
                document_path = linked_document_path / f"{document_id}.json"
                linked_documents.append(Document.load(document_path))
            if linked_documents:
                return linked_documents, _cached_parsing_stats(documents)

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

            return documents, linker.document_parsing_stats

    else:
        message = f"context type {config.default_context_type} not supported"
        fail_with_message(message)

    return None, {}


def run_extraction_pipeline(  # noqa: PLR0913
    deet_project: DeetProject,
    prompt_csv_path: Path | None,
    config_path: Path | None = None,
    prompt_population: (
        CustomPromptPopulationMethod | None
    ) = CustomPromptPopulationMethod.FILE,
    run_name: str = "",
    *,
    ignore_references: bool = False,
) -> tuple[
    ExtractionRunOutput,
    ProcessedAnnotationData,
    ExperimentArtefacts,
    DataExtractionConfig,
]:
    """Run the standard data extraction pipeline from the CLI."""
    pipeline_start = time.perf_counter()
    stage_durations: dict[str, float] = {}

    with measure_elapsed() as stage_timer:
        processed_annotation_data = deet_project.process_data()
    stage_durations[ExtractionPipelineStage.annotation_conversion] = stage_timer.seconds

    config = load_or_init_config(config_path)

    experiment_artefacts = ExperimentArtefacts.create(
        deet_project.experiments_dir, run_name=run_name
    )

    with measure_elapsed() as stage_timer:
        if prompt_population is not None:
            if (
                prompt_population == CustomPromptPopulationMethod.FILE
                and prompt_csv_path is not None
                and not prompt_csv_path.exists()
            ):
                fail_with_message(f"Prompt csv {prompt_csv_path} cannot be found")
            processed_annotation_data.populate_custom_prompts(
                method=prompt_population,
                filepath=prompt_csv_path or deet_project.prompt_csv_path,
                config=config,
            )
            if not processed_annotation_data.attributes:
                fail_with_message(
                    "No attributes selected. "
                    "Perhaps you forgot to edit your prompt file"
                )
    stage_durations[ExtractionPipelineStage.prompt_population] = stage_timer.seconds

    if not processed_annotation_data.documents:
        no_documents = "No documents found in project"
        fail_with_message(no_documents)

    evaluation_strategy = deet_project.load_evaluation_strategy()
    evaluation_strategy.snapshot(experiment_artefacts)
    active_ids = evaluation_strategy.get_active_ids(deet_project)
    processed_annotation_data.filter_documents_by_ids(active_ids)
    if not processed_annotation_data.documents:
        no_documents_in_stage = (
            "No documents in evaluation stage"
            f" '{evaluation_strategy.splits.current_stage}'."
            "\n\nManage evaluation splits with `deet experiments splits`"
        )
        fail_with_message(no_documents_in_stage)

    data_extractor = get_data_extractor(config=config)

    document_parsing: dict[str, DocumentParsingStats] = {}
    with measure_elapsed() as stage_timer:
        if ignore_references:
            if deet_project.pdf_dir_abspath is None:
                fail_with_message(
                    "This project doesn't specify a pdf directory. "
                    "Either edit the yaml file to create one or "
                    "re-initialise the project."
                )
            documents, document_parsing = create_documents_from_directory(
                deet_project.pdf_dir_abspath
            )
        else:
            documents, document_parsing = prepare_documents(
                processed_annotation_data.documents,
                config,
                linked_document_path=deet_project.linked_documents_path,
                pdf_dir=deet_project.pdf_dir_abspath,
                link_map_path=deet_project.link_map_path,
            )
    stage_durations[ExtractionPipelineStage.document_preparation] = stage_timer.seconds

    with measure_elapsed() as stage_timer:
        run_output = data_extractor.extract_from_documents(
            attributes=processed_annotation_data.attributes,
            documents=documents,
            context_type=data_extractor.config.default_context_type,
            output_file=experiment_artefacts.llm_annotations,
            document_parsing=document_parsing,
            show_progress=True,
        )
    stage_durations[ExtractionPipelineStage.llm_extraction] = stage_timer.seconds

    with measure_elapsed() as stage_timer:
        processed_annotation_data.export_attributes_csv_file(
            experiment_artefacts.prompts_snapshot
        )

        experiment_artefacts.config_snapshot.write_text(
            yaml.safe_dump(
                data_extractor.config.model_dump(mode="json"), sort_keys=False
            ),
            encoding="utf-8",
        )
    stage_durations[ExtractionPipelineStage.artifact_export] = stage_timer.seconds

    run_output.metadata.total_pipeline_duration_seconds = round(
        time.perf_counter() - pipeline_start, 3
    )
    run_output.metadata.stage_durations_seconds = stage_durations

    experiment_artefacts.extraction_metadata.write_text(
        run_output.metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )

    logger.info(f"Run metadata saved to: {experiment_artefacts.extraction_metadata}")

    return run_output, processed_annotation_data, experiment_artefacts, config


def evaluate_extraction_pipeline(
    processed_annotation_data: ProcessedAnnotationData,
    run_output: ExtractionRunOutput,
    experiment_artefacts: ExperimentArtefacts,
) -> None:
    """Evaluate results of an extraction pipeline."""
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_annotation_data.annotated_documents,
        llm_annotated_documents=run_output.annotated_documents,
        attributes=processed_annotation_data.attributes,
        extraction_run_id=experiment_artefacts.run_id,
    )
    evaluator.evaluate_llm_annotations()
    evaluator.write_metrics_to_csv(experiment_artefacts.metrics)
    evaluator.export_llm_comparison(experiment_artefacts.comparison)
    evaluator.display_metrics()
