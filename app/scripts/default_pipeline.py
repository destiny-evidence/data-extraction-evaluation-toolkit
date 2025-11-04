"""Exploration of what a pipeline might look like."""

from pathlib import Path

from loguru import logger

from app.data_models.pipeline import (
    Executor,
    Job,
    JobFormat,
    JobType,
    Language,
    ScriptExecutor,
    jobify,
)
from app.processors.parser import DocumentParser

parser = DocumentParser()


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
    ingress_method="file",
    egress_method="file",
    job=gold_standard_script_path,
    script_args=[
        "-i",
        str(gold_standard_input_data),
        "-o",
        str(gold_standard_output_dir),
    ],
    executor=Executor(executor=ScriptExecutor()),
)


# alternative version for stage 2


# default_pipeline = Pipeline(stages=[parse, translate_eppi, extract, write_stats])

# default_pipeline.execute()


# default_pipeline.extend(
#     PipelineStage("my_script.R"), job_type="validation"
# )  # add at the end
# default_pipeline.insert(PipelineStage(type="CODE"), index=1)

# default_pipeline.analyse()  # benchmarking, performance analysis

# # default_pipeline.replace(index=0, my_new_parser)

# # class Job(BaseModel):
# #     """The attributes describing a specific job."""

# #     name: str
# #     job_format: JobFormat
# #     job_type: JobType | list[JobType]
# #     language: Language
# #     ingress_method: IngressMethod | None  # we may have a job that starts with no data
# #     egress_method: EgressMethod


# my_new_parser = ParserLibrary()


# new_parser_job = Job(
#     name="parse_txt",
#     job_format="code",
#     job_type="data_processing",
#     language="python",
#     job=my_new_parser,
# )

# new_pipeline = Pipeline(PipelineStage(new))
# new_pipeline.execute()
# new_pipeline.execute()
# new_pipeline.execute()
#     language="python",
#     job=my_new_parser,
# )

# new_pipeline = Pipeline(PipelineStage(new))
# new_pipeline.execute()
# new_pipeline.execute()
# new_pipeline.execute()
#     language="python",
#     job=my_new_parser,
# )

# new_pipeline = Pipeline(PipelineStage(new))
# new_pipeline.execute()
# new_pipeline.execute()
# new_pipeline.execute()
