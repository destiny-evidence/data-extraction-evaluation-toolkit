"""Base ABC for evaluation strategies and splits objects."""

from __future__ import annotations  # Makes all type annotations lazy strings

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
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

    @abstractmethod
    def get_active_ids(self, project: DeetProject) -> list[int]:
        """Return IDs to run the pipeline on."""

    @abstractmethod
    def snapshot(self, artefacts: ExperimentArtefacts) -> None:
        """Persist stategy state alongside an experiment run."""

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

    current_stage: StageT

    def dump_to_json(self, path: Path) -> None:
        """Persist split state to json."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @property
    @abstractmethod
    def active_ids(self) -> list[int]:
        """Return the current active ids."""
