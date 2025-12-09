"""Command line script to run a pipeline on a batch of records in a project."""

import json
from pathlib import Path

import typer

from deet.data_models.base import Attribute, Document, GoldStandardAnnotation
from deet.data_models.eppi import EppiAttribute
from deet.data_models.pipeline import JobType, Pipeline, jobify, stage_from_job
from deet.data_models.project import DeetProject
from deet.extractors.llm_data_extractor import (
    ContextType,
    DataExtractionConfig,
    LLMDataExtractor,
)

# NOTE - define your LLM config stuff here. currently all values are default.
config = DataExtractionConfig(context_type=ContextType.ABSTRACT_ONLY)

data_extractor = LLMDataExtractor(config=config)


def llm_data_extraction(
    documents: list[Document],
    attributes_file_path: Path,
    output_path: Path,
    filter_by_attribute_ids: list[int] | None = None,
    **kwargs,
) -> list[GoldStandardAnnotation]:
    """Run LLM data extraction."""
    attributes_raw = json.loads(attributes_file_path.read_text())

    attributes: list[Attribute] = [EppiAttribute(**record) for record in attributes_raw]
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


app = typer.Typer(help="Create a DEET project")


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

    llm_extraction_stage = stage_from_job(
        jobify(
            name="llm_extraction",
            job_type=JobType.EXTRACTION,
            func_kwargs={
                "documents": proj.read_annotated_documents(
                    batch_ids=proj.documents_in_batches
                ),
                "attributes_file_path": proj.proc_data / "attributes_filtered.json",
                "output_path": run_path / "llm_extractions.json",
                "prompt_outfile": run_path / "full_prompt_payload.json",
            },
        )(llm_data_extraction)
    )

    my_beautiful_pipeline = Pipeline(
        name="test_pipeline",
        stages=[llm_extraction_stage],
        # stages=[ingest_gs_stage, llm_extraction_stage],
    )

    my_beautiful_pipeline.run()


def main() -> None:
    """Run the typer app."""
    app()


if __name__ == "__main__":
    main()
