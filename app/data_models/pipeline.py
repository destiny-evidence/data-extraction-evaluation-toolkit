"""Models to employ for implementing DEET jobd in sequential, harmonised _pipelines_."""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class IngressMethod(StrEnum):
    """An enum of ingress methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()
    HTTP = auto()  # we may need to download data
    RANDOM = auto()  # there might be jobs & pipeline stages where we start with a seed


class EgressMethod(StrEnum):
    """An enum of egree methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()


class JobFormat(StrEnum):
    """
    An enum of job formats.

    Jobs are the building blocks of pipeline stages.
    The job format describes the 'medium' in which the job
    is provided to the job object.
    """

    SCRIPT = auto()  # when job is supplied in a file.
    CODE = auto()  # when job is supplied within the pipeline.


class JobType(StrEnum):
    """
    An enum of job types.

    This is a descriptive label of the broad category
    of what the job is doing.
    """

    DATA_PROCESSING = auto()
    DATA_COLLECTION = auto()
    CLASSIFICATION = auto()
    EXTRACTION = auto()


class Language(StrEnum):
    """An enum of permitted languages a job can be specified in."""

    PYTHON = auto()
    R = auto()
    SHELL = auto()
    SQL = auto()
    LLM_PROMPT = auto()


class JobExecutor:
    """A wrapper around various tools for executing a job given job configuration."""

    def __init__(self, language: Language, job_format: JobFormat):
        pass


class DataFormat(BaseModel):
    """
    The format data at a given stage of a job (ingress or egress).

    This is regardless of whether it is passed
    along from a previous stage in memory, or
    read from a file.

    This shouldn't represent a basic built-in
    like `str`, but rather reflect a specific str
    format like markdown.
    """

    data_type: type
    name: str | None


class Job(BaseModel):
    """The attributes describing a specific job."""

    name: str
    job_format: JobFormat
    job_type: JobType | list[JobType]
    language: Language
    ingress_method: IngressMethod | None  # we may have a job that starts with no data
    egress_method: EgressMethod


class PipelineStage(BaseModel):
    """A stage in a DEET pipeline."""

    name: str
    skip_if_failed: bool = True
    permitted_ingress_methods = list[IngressMethod]
    permitted_egress_methods = list[EgressMethod]
    input_file: Path | None
    data: Any | None
    jobs: Job | list[Job]

    @classmethod
    @field_validator("jobs", mode="before")
    def convert_jobs_to_list(cls, v: Job | list[Job]) -> list[Job]:
        """Convert jobs to list of jobs if just one job supplied."""
        if isinstance(v, Job):
            v = [v]
        return v

    @classmethod
    @field_validator
    def read_if_file(cls, v) -> None:
        return


class Pipeline(BaseModel):
    """A complete pipeline consisting of several `PipelineStage` objects."""

    name: str
    stages: list[PipelineStage]
