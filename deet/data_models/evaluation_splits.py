"""Data models for to split documents into development, validation, and test sets."""

import random
from collections.abc import Collection
from enum import StrEnum, auto
from pathlib import Path

from pydantic import BaseModel, Field

from deet.exceptions import SplitsValidationError


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

    @property
    def active_ids(self) -> list[int]:
        """Return the list of document IDs for the current evaluation stage."""
        stage_mapping = {
            EvaluationStage.DEVELOPMENT: self.development_ids,
            EvaluationStage.VALIDATION: self.validation_ids,
            EvaluationStage.TEST: self.test_ids,
        }
        return stage_mapping[self.current_stage]

    def get_unassigned_ids(self, project_doc_ids: Collection[int]) -> list[int]:
        """Filter a collection of document IDs to those which have not been assigned."""
        assigned = (
            set(self.development_ids) | set(self.validation_ids) | set(self.test_ids)
        )
        return [doc_id for doc_id in project_doc_ids if doc_id not in assigned]

    def add_to_development(self, project_doc_ids: Collection[int], size: int) -> int:
        """Sample from project_doc_ids and add to development."""
        self.current_stage = EvaluationStage.DEVELOPMENT
        unassigned = self.get_unassigned_ids(project_doc_ids)

        if size <= 0:
            too_small = "Sample size must be greater than 0."
            raise SplitsValidationError(too_small)
        if len(unassigned) < size:
            incompatible_size = (
                f"Tried to assign {size} docs to the development set"
                f" but only {len(unassigned)} are unassigned"
            )
            raise SplitsValidationError(incompatible_size)
        target_ids = random.sample(unassigned, size)

        self.development_ids.extend(target_ids)
        return len(target_ids)
