"""Exploration of what a pipeline might look like."""

import json
from pathlib import Path

from app.data_models.eppi import EppiAttribute, EppiDocument
from app.data_models.pipeline import (
    EgressMethod,
    Executor,
    IngressMethod,
    Job,
    JobFormat,
    JobType,
    Language,
    Pipeline,
    ScriptExecutor,
    jobify,
    stage_from_job,
)
from app.extractors.data_extractor import DataExtractionConfig, DataExtractor
from app.processors.eppi_annotation_converter import (
    EppiAnnotationConverter,
    EppiGoldStandardAnnotation,
)
from app.processors.parser import DocumentParser

parser = DocumentParser()
converter = EppiAnnotationConverter()

config = DataExtractionConfig()

data_extractor = DataExtractor(config=config)


# stage 1 -- parse pdf
@stage_from_job
@jobify(name="parse_pdf", job_type=JobType.DATA_PROCESSING)
def parse_pdf(
    pdf_path: Path = Path("misc/test_data_simplified/Jenner_2022.pdf"),
    out_path: Path = Path("misc/test_data_simplified/Jenner_2022.md"),
) -> None:
    """Parse pdf to Markdown."""
    parser(input_=pdf_path, out_path=out_path)


# stage 2 -- example for a script!
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
    func_kwargs={
        "eppi_json_path": Path(
            "misc/test_data_simplified/Jenner_2022_ground_truth.json"
        ),
        "output_dir": Path("misc/test_data_simplified/converted_data_models"),
    },
)
def ingest_gold_standard_func(eppi_json_path: Path, output_dir: Path) -> None:
    """Convert EPPI JSON to DEET data models."""
    out = converter.process_annotation_file(eppi_json_path)
    converter.save_processed_data(processed_data=out, output_dir=output_dir)


# stage 3, extract data
@stage_from_job
@jobify(
    name="llm_data_extraction",
    job_type=JobType.EXTRACTION,
    func_kwargs={
        "full_text_path": Path("misc/test_data_simplified/Jenner_2022.md"),
        "documents_file_path": Path(
            "misc/test_data_simplified/converted_data_models/eppi/Jenner_2022_ground_truth/documents.json"
        ),
        "attributes_file_path": Path(
            "misc/test_data_simplified/converted_data_models/eppi/Jenner_2022_ground_truth/attributes.json"
        ),
        "output_path": Path("misc/test_data_simplified/llm_extraction_test.json"),
    },
)
def llm_data_extraction(
    full_text_path: Path,
    documents_file_path: Path,
    attributes_file_path: Path,
    output_path: Path,
) -> list[EppiGoldStandardAnnotation]:
    """Run LLM data extraction."""
    full_text = full_text_path.read_text()

    documents_raw = json.loads(documents_file_path.read_text())
    attributes_raw = json.loads(attributes_file_path.read_text())

    attributes = [EppiAttribute(**record) for record in attributes_raw]
    documents = [EppiDocument(**record) for record in documents_raw]

    return data_extractor.extract_from_documents(
        documents=documents,
        attributes=attributes,
        output_file=output_path,
        full_text=full_text,
    )


# my_beautiful_pipeline = Pipeline(
#     name="test_pipeline",
#     stages=[parse_pdf, ingest_gold_standard_func, llm_data_extraction],
# )
my_beautiful_pipeline = Pipeline(
    name="test_pipeline",
    stages=[
        ingest_gold_standard_func,
        llm_data_extraction,
    ],  # assumes we already have the parsed pdf.
)


if __name__ == "__main__":
    my_beautiful_pipeline.run()


# we can now run this using
# >>> ingest_gold_standard_func.run_jobs()
# my_pipeline = Pipeline(stages=[parse_pdf, ingest_gold_standard, run_llm_extraction])
