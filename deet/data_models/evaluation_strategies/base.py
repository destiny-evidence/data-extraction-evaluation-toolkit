"""
Evaluation strategy framework: dividing documents between stages for evaluation.

This module defines base classes for managing how documents are partitioned across
evaluation stages. This module provides three abstractions:

- `BaseEvaluationStage`: An enum of the concrete stages for a strategy.
- `BaseSplits`: A Pydantic model persisting the partition of documents
  across stages and the current stage of an experiment.
- `BaseEvaluationStrategy`: The orchestrator, managing splits and providing
  interactive and programmatic interfaces for moving documents between stages.

Concrete strategies implement these abstractions and are registered
in the package `__init__.py`.
"""

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
    """
    Defines the stages of an evaluation strategy.

    Each of these stages is mapped to a list of IDs in the splits model below.
    """


class BaseSplits[StageT: BaseEvaluationStage](BaseModel):
    """
    Base object for persisting how document IDs are used for an experiment.

    An evaluation strategy divides a project's documents into stages
    (e.g. dev, validation, and test). This model stores that partition,
    alongside the stage the experiment is currently in, and serialises to/from
    JSON.

    Subclasses define the concrete shape and must:
    - declare one model field per stage
    - bind `StageT` to the strategy's stage enum
    - set the `_STAGE_FIELD_NAMES` class var mapping each stage in the enum
        to its model field

    This contract is enforced on definition, and ensures that generic methods
    can be defined here for DRYness.
    """

    _STAGE_FIELD_NAMES: ClassVar[dict[BaseEvaluationStage, str]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        """Validate whether a subclass honours contract defined by this base class."""
        super().__pydantic_init_subclass__(**kwargs)

        if not hasattr(cls, "_STAGE_FIELD_NAMES"):
            undeclared = f"{cls.__name__} must define _STAGE_FIELD_NAMES"
            raise TypeError(undeclared)

        undefined = [
            field
            for field in cls._STAGE_FIELD_NAMES.values()
            if field not in cls.model_fields
        ]
        if undefined:
            bad_fields = (
                f"{cls.__name__}._STAGE_FIELD_NAMES maps to undefined"
                f" model fields(s): {undefined}"
            )
            raise TypeError(bad_fields)

        stage_enum = cls.model_fields["current_stage"].annotation
        if isinstance(stage_enum, type) and issubclass(stage_enum, BaseEvaluationStage):
            mapped = set(cls._STAGE_FIELD_NAMES)
            expected: set[BaseEvaluationStage] = set(stage_enum)
            if mapped != expected:
                mismatch = (
                    f"{cls.__name__}._STAGE_FIELD_NAMES keys must match "
                    f"{stage_enum.__name__} exactly; "
                    f"missing={expected - mapped}, unexpected={mapped - expected}"
                )
                raise TypeError(mismatch)

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


class BaseEvaluationStrategy[SplitsT: BaseSplits](ABC):
    """
    Orchestrate how a project's documents are divided into evaluation stages.

    An evaluation strategy defines stages (via a `BaseEvaluationStage` enum)
    and how to move documents through them.

    When `deet.extractors.cli_helpers.run_extraction_pipeline` is called, it
    filters documents down to those that are assigned to the current stage.

    Each subclass implements its own methods for managing splits, but each
    must implement a `run_splits_wizard` method, which is called when a user
    runs `deet experiments splits`.
    """

    name: EvaluationStrategyName

    def __init__(self, project: DeetProject) -> None:
        """Initialise and set splits."""
        self._project = project
        self.splits = self._load_splits(project)

    @abstractmethod
    def _load_splits(self, project: DeetProject) -> SplitsT:
        """Load or init the splits model for this strategy and project."""

    def get_active_ids(self, project: DeetProject) -> list[int]:
        """Return the document IDs in the current stage to run the pipeline on."""
        return self.splits.active_ids

    def snapshot(self, artefacts: ExperimentArtefacts) -> None:
        """Persist stategy state with an experiment run's artefacts."""
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
        """
        Run the interactive workflow for moving documents between stages.

        By default (``action=None``), presents an interactive prompt listing the
        actions available in the current stage and strategy, and the user selects
        one. Optionally, ``--action`` can be passed to skip the prompt (for
        scripted use). Unknown or stage-inappropriate actions raise an error rather
        than silently failing.

        Further details (e.g. how many documents to move) are prompted for
        interactively if not provided (``size``, ``experiment``). Subclasses
        implement the concrete stage transitions and validation logic.

        Note:
            The valid actions depend on both the strategy and the current stage,
            which are only known at runtime. Static CLI subcommands would
            misleadingly list all actions in ``--help`` regardless of whether they
            apply to the project's configured strategy and current stage. Instead,
            the action is discovered interactively (the prompt shows what's valid
            *now*) or specified as an unconstrained string for scripted use.

        """
