"""Base ABC for evaluation strategies and splits objects."""

from __future__ import annotations  # Makes all type annotations lazy strings

import random
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from deet.exceptions import SplitsValidationError

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from deet.data_models.enums import EvaluationStrategyName
    from deet.data_models.project import DeetProject, ExperimentArtefacts


class BaseEvaluationStage(StrEnum):
    """Defines the stages of an evaluation strategy."""


class BaseEvaluationStrategy[SplitsT: BaseSplits](ABC):
    """Base class defining methods of an evaluation strategy."""

    name: EvaluationStrategyName

    def __init__(self, project: DeetProject) -> None:
        """initialise and set splits."""
        self._project = project
        self.splits = self._load_splits(project)

    @abstractmethod
    def _load_splits(self, project: DeetProject) -> SplitsT:
        """Return the splits object for this strategy and project."""

    def get_active_ids(self, project: DeetProject) -> list[int]:
        """Return IDs to run the pipeline on."""
        return self.splits.active_ids

    def snapshot(self, artefacts: ExperimentArtefacts) -> None:
        """Persist stategy state alongside an experiment run."""
        self.splits.dump_to_json(artefacts.evaluation_splits_snapshot)

    @abstractmethod
    def run_splits_wizard(
        self,
        project: DeetProject,
        *,
        action: str | None = None,
        size: int | None = None,
        experiment: str | None = None,
    ) -> None:
        """Manage the interactive splits workflow."""


class BaseSplits[StageT: BaseEvaluationStage](BaseModel):
    """Base object for persisting how document IDs are used for an experiment."""

    _STAGE_FIELD_NAMES: ClassVar[dict[BaseEvaluationStage, str]]

    def __init_subclass__(cls, **kwargs) -> None:
        """Check whether subclasses have defined _STAGE_FIELD_NAMES."""
        if "_STAGE_FIELD_NAMES" not in cls.__dict__:
            missing_names = f"{cls.__name__} must define _STAGE_FIELD_NAMES"
            raise TypeError(missing_names)
        super().__init_subclass__(**kwargs)

    current_stage: StageT

    def dump_to_json(self, path: Path) -> None:
        """Persist split state to json."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def _get_list_for_stage(self, stage: StageT) -> list[int]:
        """Get the the ids in the field corresponding the stage."""
        return getattr(self, self._STAGE_FIELD_NAMES[stage])

    def get_unassigned_ids(self, project_doc_ids: Collection[int]) -> list[int]:
        """Filter a collection of document IDs to those which have not been assigned."""
        assigned = set()
        for field_name in self._STAGE_FIELD_NAMES.values():
            assigned.update(getattr(self, field_name))

        return [doc_id for doc_id in project_doc_ids if doc_id not in assigned]

    def add_to_stage(
        self,
        stage: StageT,
        project_doc_ids: Collection[int],
        size: int,
    ) -> int:
        """Sample from unassigned and add to a stage."""
        unassigned = self.get_unassigned_ids(project_doc_ids)
        target_list = self._get_list_for_stage(stage)

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
        target_list.extend(target_ids)
        return len(target_ids)

    @property
    def active_ids(self) -> list[int]:
        """Return the current active ids, based on the current stage."""
        return self._get_list_for_stage(self.current_stage)
