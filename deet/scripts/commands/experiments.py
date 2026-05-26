# ruff: noqa: PLC0415
"""CLI sub-commands for running data extraction experiments (and evaluating them)."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from deet.data_models.enums import CustomPromptPopulationMethod
from deet.exceptions import SplitsValidationError
from deet.scripts.typer_context import project_required

if TYPE_CHECKING:
    from deet.data_models.project import DeetProject

app = typer.Typer(
    help=(
        "Commands to create and evaluate data extraction "
        "experiments within your project."
    )
)


@app.command()
@project_required
def evaluate(
    typer_context: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to a config file containing options for data "
            "extraction configuration. A template with defaults is generated"
            " on project setup."
            "\nLeave this blank to configure interactively."
        ),
    ] = None,
    prompt_population: Annotated[
        CustomPromptPopulationMethod | None,
        typer.Option(
            help="A method to define custom prompts for your attributes to be "
            "extracted. Leave blank to use the prompts in your gold standard "
            "data. Set to `file` to provide a file of prompt definitions "
            "(make sure this is supplied below). Set to `cli` to define prompts"
            " interactively in the CLI. With `file`, only attributes that appear "
            "in the CSV with a non-empty `prompt` are kept for extraction and "
            "evaluation (see also `--csv-path`)."
        ),
    ] = CustomPromptPopulationMethod.FILE,
    run_name: Annotated[
        str,
        typer.Option(
            help="A name for the run (which will appended to a timestamp) "
            "to help you identify this run later"
        ),
    ] = "",
    custom_evaluation_metrics: Annotated[
        list[str] | None,
        typer.Option(
            help="A list of additional sklearn metrics that you wish to "
            " calculate. Use this option for each additional metric you "
            " would like to add, e.g. `deet extract-data "
            "--custom-evaluation-metrics brier_score_loss "
            "--custom-evaluation-metrics cohen_kappa_score`"
        ),
    ] = None,
) -> None:
    """
    Extract data from documents and evaluate.

    Load gold standard annotation data, and use an LLM to extract data from the
    documents in your dataset. Evaluate by comparing the results to the gold
    standard data.
    """
    from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
    from deet.extractors.cli_helpers import run_extraction_pipeline

    run_output, processed_annotation_data, experiment_artefacts = (
        run_extraction_pipeline(
            typer_context=typer_context,
            config_path=config_path,
            prompt_population=prompt_population,
            run_name=run_name,
        )
    )

    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_annotation_data.annotated_documents,
        llm_annotated_documents=run_output.annotated_documents,
        attributes=processed_annotation_data.attributes,
        custom_metrics=custom_evaluation_metrics,
        extraction_run_id=experiment_artefacts.run_id,
    )
    evaluator.evaluate_llm_annotations()
    evaluator.write_metrics_to_csv(experiment_artefacts.metrics)
    evaluator.export_llm_comparison(experiment_artefacts.comparison)
    evaluator.display_metrics()


@app.command()
@project_required
def predict(
    typer_context: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to a config file containing options for data "
            "extraction configuration. A template with defaults is generated"
            " on project setup."
            "\nLeave this blank to configure interactively."
        ),
    ] = None,
    prompt_population: Annotated[
        CustomPromptPopulationMethod | None,
        typer.Option(
            help="A method to define custom prompts for your attributes to be "
            "extracted. Leave blank to use the prompts in your gold standard "
            "data. Set to `file` to provide a file of prompt definitions "
            "(make sure this is supplied below). Set to `cli` to define prompts"
            " interactively in the CLI. With `file`, only attributes that appear "
            "in the CSV with a non-empty `prompt` are kept for extraction and "
            "evaluation (see also `--csv-path`)."
        ),
    ] = CustomPromptPopulationMethod.FILE,
    run_name: Annotated[
        str,
        typer.Option(
            help="A name for the run (which will appended to a timestamp) "
            "to help you identify this run later"
        ),
    ] = "",
    ignore_references: bool = typer.Option(  # noqa: FBT001
        default=False,
        help=(
            "Ignore references in gold standard data and just"
            "extract from whatever is in your pdf_dir"
        ),
    ),
) -> None:
    """
    Extract data from documents without evaluating.

    Load gold standard annotation data, and use an LLM to extract data from the
    documents in your dataset. When used with ignore_references = True,
    documents are created directly from the files contained in pdf_dir.
    """
    from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
    from deet.extractors.cli_helpers import run_extraction_pipeline

    (run_output, processed_annotation_data, experiment_artefacts) = (
        run_extraction_pipeline(
            typer_context=typer_context,
            config_path=config_path,
            prompt_population=prompt_population,
            run_name=run_name,
            ignore_references=ignore_references,
        )
    )

    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=[],
        llm_annotated_documents=run_output.annotated_documents,
        attributes=processed_annotation_data.attributes,
        extraction_run_id=experiment_artefacts.run_id,
    )
    evaluator.export_llm_csv(experiment_artefacts.llm_annotation_csv)


@app.command()
@project_required
def add_dev(
    typer_context: typer.Context,
    size: Annotated[
        int,
        typer.Option(
            "--size",
            "-s",
            help="Number of random unassigned documents to sample and add.",
        ),
    ],
) -> None:
    """Add unassigned documents to the development pool."""
    from deet.data_models.evaluation_splits import EvaluationStage
    from deet.ui import fail_with_message, notify

    deet_project: DeetProject = typer_context.obj.project
    project_doc_ids = deet_project.get_all_doc_ids()

    splits = deet_project.load_splits()
    try:
        n_added = splits.add_to_stage(
            EvaluationStage.DEVELOPMENT, project_doc_ids, size
        )
        splits.dump_to_json(deet_project.evaluation_splits_path)

    except SplitsValidationError as e:
        fail_with_message(str(e))

    notify(
        f"Added {n_added} documents to development set."
        f" This now contains {len(splits.development_ids)} documents."
        f" {len(splits.get_unassigned_ids(project_doc_ids))}"
        " are still unassigned."
    )


@app.command()
@project_required
def validate_run(
    typer_context: typer.Context,
    size: Annotated[
        int,
        typer.Option(
            "--size",
            "-s",
            help="Number of random unassigned documents to sample and add.",
        ),
    ],
) -> None:
    """Select a past experiment config and evaluate against a fresh validation set."""
    from InquirerPy import inquirer

    from deet.data_models.evaluation_splits import EvaluationStage
    from deet.data_models.project import ExperimentArtefacts
    from deet.extractors.cli_helpers import (
        evaluate_extraction_pipeline,
        run_extraction_pipeline,
    )
    from deet.ui import fail_with_message, notify

    deet_project: DeetProject = typer_context.obj.project

    all_experiments = [
        ExperimentArtefacts(base_dir=path)
        for path in deet_project.experiments_dir.iterdir()
        if path.is_dir()
    ]
    completed_experiments = [exp for exp in all_experiments if exp.is_complete]
    completed_experiments.sort(key=lambda e: e.run_id, reverse=True)

    choices = [{"name": exp.run_id, "value": exp} for exp in completed_experiments]

    selected_experiment: ExperimentArtefacts = inquirer.select(
        message="Select the experiment configuration to validate:", choices=choices
    ).execute()

    project_doc_ids = deet_project.get_all_doc_ids()

    splits = deet_project.load_splits()
    try:
        n_added = splits.add_to_stage(EvaluationStage.VALIDATION, project_doc_ids, size)
        splits.current_stage = EvaluationStage.VALIDATION
        splits.dump_to_json(deet_project.evaluation_splits_path)
    except SplitsValidationError as e:
        fail_with_message(str(e))

    notify(
        f"Added {n_added} documents to validation set"
        f" ({len(splits.get_unassigned_ids(project_doc_ids))}"
        " are still unassigned)."
        f"\nEvaluating experiment: {selected_experiment.run_id} using these documents"
    )

    run_output, processed_annotation_data, experiment_artefacts = (
        run_extraction_pipeline(
            typer_context=typer_context,
            config_path=selected_experiment.config_snapshot,
            run_name="VALIDATION",
        )
    )
    evaluate_extraction_pipeline(
        processed_annotation_data=processed_annotation_data,
        run_output=run_output,
        experiment_artefacts=experiment_artefacts,
    )

    decision = inquirer.select(
        message="Based on these metrics, how would you like to proceed?",
        choices=[
            {
                "name": (
                    "Accept: lock this configuration, and "
                    " do a final test on all remaining documents."
                ),
                "value": "accept",
            },
            {
                "name": (
                    "Reject: add validation documents to "
                    " development set and continue iterating."
                ),
                "value": "reject",
            },
        ],
    ).execute()

    if decision == "accept":
        try:
            splits.finalise_test(project_doc_ids)
        except SplitsValidationError as e:
            fail_with_message(str(e))

        splits.dump_to_json(deet_project.evaluation_splits_path)

        run_output, processed_annotation_data, experiment_artefacts = (
            run_extraction_pipeline(
                typer_context=typer_context,
                config_path=selected_experiment.config_snapshot,
                run_name="FINAL_TEST",
            )
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed_annotation_data,
            run_output=run_output,
            experiment_artefacts=experiment_artefacts,
        )

    elif decision == "reject":
        splits.reject_validation()
        splits.dump_to_json(deet_project.evaluation_splits_path)
