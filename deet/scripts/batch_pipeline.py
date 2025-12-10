"""Command line script to run a pipeline on a batch of records in a project."""

from pathlib import Path

import typer
import yaml

from deet.data_models.base import (
    Attribute,
    Document,
    GoldStandardAnnotatedDocument,
    ProcessedAttributeData,
)
from deet.data_models.pipeline import JobType, Pipeline, jobify, stage_from_job
from deet.data_models.project import DeetProject
from deet.extractors.llm_data_extractor import (
    DataExtractionConfig,
    LLMDataExtractor,
)


def llm_data_extraction(
    documents: list[Document],
    attributes: list[Attribute],
    output_path: Path,
    data_extractor: LLMDataExtractor,
    filter_by_attribute_ids: list[int] | None = None,
    **kwargs,
) -> list[GoldStandardAnnotatedDocument]:
    """Run LLM data extraction."""
    if filter_by_attribute_ids:
        attributes = [
            a for a in attributes if a.attribute_id in filter_by_attribute_ids
        ]

    return data_extractor.extract_from_documents(
        documents=documents,
        attributes=attributes,
        output_file=output_path,
        **kwargs,
    )


app = typer.Typer(help="Run a pipeline on your batch")


@app.command()
def batch_pipeline() -> None:
    """Run a pipeline on a batch of documents."""
    proj = DeetProject(path=".")

    if len(proj.batches) == 0:
        typer.echo(
            "No batches in this project. Create one first with DEET-new-batch --help"
        )
        raise typer.Abort()  # noqa: RSE102

    runs = list(proj.batches[-1].iterdir())
    run_path = proj.batches[-1] / f"run_{len(runs)}"
    run_path.mkdir()

    run_config = yaml.safe_load(proj.p.joinpath("run-settings.yaml").open())

    config = DataExtractionConfig(**run_config)

    data_extractor = LLMDataExtractor(config=config)

    run_config_path = run_path / "run_settings.json"

    run_config_path.write_text(config.model_dump_json())

    if config.prompt_csv_path is not None:
        proc_attributes = ProcessedAttributeData(attributes=proj.read_attributes())
        prompt_csv_path = proj.p / "prompts" / config.prompt_csv_path
        proc_attributes.populate_custom_prompts(method="file", filepath=prompt_csv_path)

        attributes = proc_attributes.attributes
    else:
        attributes = proj.read_attributes()

    llm_extraction_stage = stage_from_job(
        jobify(
            name="llm_extraction",
            job_type=JobType.EXTRACTION,
            func_kwargs={
                "documents": proj.read_annotated_documents(
                    batch_ids=proj.documents_in_batches
                ),
                "attributes": attributes,
                "output_path": run_path / "llm_extractions.json",
                "prompt_outfile": run_path / "full_prompt_payload.json",
                "data_extractor": data_extractor,
            },
        )(llm_data_extraction)
    )

    my_beautiful_pipeline = Pipeline(
        name="test_pipeline",
        stages=[llm_extraction_stage],
        # stages=[ingest_gs_stage, llm_extraction_stage],
    )

    my_beautiful_pipeline.run()

    proj.evaluate_run(run_path)


def main() -> None:
    """Run the typer app."""
    app()


if __name__ == "__main__":
    main()
