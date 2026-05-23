"""Data models for to split documents into development, validation, and test sets."""

from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationStage(StrEnum):
    """
    Types of evaluation (see https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data).

    DEVELOPMENT is used to iterate and improve prompts/configuration.
    VALIDATION is used to validate prompts on data they have not been tuned for.
    TEST is used for a final assessment.
    """

    DEVELOPMENT = auto()
    VALIDATION = auto()
    TEST = auto()


class EvaluationSplits(BaseModel):
    """Keeps track of the way documents have been assigned to stages."""

    current_stage: EvaluationStage = EvaluationStage.DEVELOPMENT
    development_ids: list[int] = Field(default_factory=list)
    validation_ids: list[int] = Field(default_factory=list)
    test_ids: list[int] = Field(default_factory=list)

    def dump_to_json(self, file_path: Path) -> None:
        """Write splits to file."""
        file_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Path) -> "EvaluationSplits":
        """Load splits from file."""
        if not file_path.exists():
            return cls()
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))
