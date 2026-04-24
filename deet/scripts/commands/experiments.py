# ruff: noqa: PLC0415
"""CLI sub-commands for running data extraction experiments (and evaluating them)."""

from pathlib import Path
from typing import Annotated

import typer

from deet.data_models.enums import CustomPromptPopulationMethod
from deet.scripts.typer_context import project_required

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
) -> None:
    """
    Extract data from documents without evaluating.

    Load gold standard annotation data, and use an LLM to extract data from the
    documents in your dataset.
    """
    from deet.extractors.cli_helpers import run_extraction_pipeline

    run_output, processed_annotation_data, experiment_artefacts = (
        run_extraction_pipeline(
            typer_context=typer_context,
            config_path=config_path,
            prompt_population=prompt_population,
            run_name=run_name,
        )
    )
