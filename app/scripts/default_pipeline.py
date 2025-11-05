"""Exploration of what a pipeline might look like."""

from pathlib import Path

from app.data_models.pipeline import (
    EgressMethod,
    Executor,
    IngressMethod,
    Job,
    JobFormat,
    JobType,
    Language,
    ScriptExecutor,
    jobify,
    stage_from_job,
)
from app.processors.eppi_annotation_converter import (
    EppiAnnotationConverter,
    ProcessedAnnotationData,
)
from app.processors.parser import DocumentParser

parser = DocumentParser()
converter = EppiAnnotationConverter()


# stage 1 -- parse pdf
@jobify(name="parse_pdf", job_type=JobType.DATA_PROCESSING)
def parse_pdf(
    pdf_path: Path = Path("misc/test_data/Abrantes_2014.pdf"),
    out_path: Path = Path("misc/parsed/Abrantes_2014.md"),
) -> None:
    """Parse pdf to Markdown."""
    parser(input_=pdf_path, out_path=out_path)


# stage 2 -- ingest gold standard data, eppi json
gold_standard_script_path = Path("app/scripts/eppi_annotation_converter_cli.py")
gold_standard_input_data = Path("misc/test_data/Abrantes_2014.json")
gold_standard_output_dir = Path("misc/test_data/gs_out/")

ingest_gold_standard = Job(
    name="ingest_gold_standard",
    job_format=JobFormat.SCRIPT,
    job_type=JobType.DATA_PROCESSING,
    language=Language.PYTHON,
    ingress_method=IngressMethod.FILE,
    egress_method=EgressMethod.FILE,
    job=gold_standard_script_path,
    script_args=[
        "-i",
        str(gold_standard_input_data),
        "-o",
        str(gold_standard_output_dir),
    ],
    executor=Executor(executor=ScriptExecutor()),
)
ingest_gold_standard_stage = stage_from_job(ingest_gold_standard)


# alternative version for stage 2
@stage_from_job  # converts a `Job` to a `PipelineStage`
@jobify(
    name="ingest_gs_func",
    func_kwargs={"eppi_json_path": Path("misc/test_data/Abrantes_2014.json")},
)
def ingest_gold_standard_func(eppi_json_path: Path) -> ProcessedAnnotationData:
    """Convert EPPI JSON to DEET data models."""
    return converter.process_annotation_file(eppi_json_path)


# we can now run this using
# >>> ingest_gold_standard_func.run_jobs()
