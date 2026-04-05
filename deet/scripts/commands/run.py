# ruff: noqa: PLC0415
"""CLI sub-commands for running data extraction experiments (and evaluating them)."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from deet.data_models.project import DeetProject


import typer

from deet.data_models.enums import CustomPromptPopulationMethod
from deet.scripts.context import project_required
from deet.ui import fail_with_message
from deet.ui.terminal import console, render_template, run_model_wizard
from deet.ui.terminal.components import info_panel
from deet.ui.terminal.wizards import continue_after_key

app = typer.Typer(help="Data extraction experiments")


@app.command()
@project_required
def extract(
    ctx: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to a config file containing options for data "
            "extraction config. A template with defaults is generated"
            " on project setup."
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
    import yaml

    from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator
    from deet.extractors.cli_helpers import (
        init_extraction_run,
        load_or_init_config,
        prepare_documents,
    )
    from deet.extractors.llm_data_extractor import (
        DataExtractionConfig,
        LLMDataExtractor,
    )

    deet_project: DeetProject = ctx.obj.project
    processed_annotation_data = deet_project.process_data()

    if config_path is None:
        console.clear()
        console.print(
            info_panel(
                render_template("extraction/config_init"),
                "Data extraction config wizard",
            )
        )
        continue_after_key()
        config = run_model_wizard(DataExtractionConfig)
    else:
        config = load_or_init_config(config_path=config_path)

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
