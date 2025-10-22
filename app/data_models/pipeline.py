"""Models to employ for implementing DEET tasks in sequential, harmonised fashion."""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class IngressMethod(StrEnum):
    """An enum of ingress methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()


class EgressMethod(StrEnum):
    """An enum of egree methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()


class InputFormat(BaseModel):
    """
    The format of incoming data.

    This is regardless of whether it is passed
    along from a previous stage in memory, or
    read from a file.

    This shouldn't represent a basic built-in
    like `str`, but rather reflect a specific str
    format like markdown.
    """

    data_type: type
    name: str | None


class PipelineStage(BaseModel):
    """A stage in a DEET pipeline."""

    name: str
    skip_if_failed: bool = True
    permitted_ingress_methods = list[IngressMethod]
    permitted_egress_methods = list[EgressMethod]
    input_file: Path | None
    data: Any | None

    @classmethod
    @field_validator
    def read_if_file(cls, v) -> None:
        return


class Pipeline(BaseModel):
    """A complete pipeline consisting of several `PipelineStage` objects."""

    name: str
    stages: list[PipelineStage]
