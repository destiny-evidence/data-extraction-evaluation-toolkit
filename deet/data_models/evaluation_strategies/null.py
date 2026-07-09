"""Null Evaluation strategy (== state prior to introduction of strategies)."""

from enum import auto
from typing import ClassVar

from deet.data_models.enums import EvaluationStrategyName
from deet.data_models.evaluation_strategies.base import (
    BaseEvaluationStage,
    BaseEvaluationStrategy,
    BaseSplits,
)
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.ui.messenger import notify


class NullStage(BaseEvaluationStage):
    """
    Stages of the null strategy.

    In this case, the only stage is ALL -> evaluate all documents.
    """

    ALL = auto()


class NullSplits(BaseSplits):
    """Records document IDs used in the null strategy."""

    _STAGE_FIELD_NAMES: ClassVar[dict[BaseEvaluationStage, str]] = {
        NullStage.ALL: "all_ids"
    }

    current_stage: NullStage = NullStage.ALL

    all_ids: list[int]


class NullEvaluationStrategy(BaseEvaluationStrategy[NullSplits]):
    """The default evaluation strategy (use all documents)."""

    name = EvaluationStrategyName.NONE

    def _load_splits(self, project: "DeetProject") -> NullSplits:
        """Make a new splits object with all project IDs."""
        return NullSplits(all_ids=project.get_all_doc_ids())

    def snapshot(self, artefacts: "ExperimentArtefacts") -> None:
        """Persist all project document ids."""
        self.splits.dump_to_json(artefacts.evaluation_splits_snapshot)

    def run_splits_wizard(
        self,
        project: DeetProject,
        *,
        action: str | None = None,
        size: int | None = None,
        experiment: str | None = None,
    ) -> None:
        """Run the splits wizard."""
        notify(
            "No evaluation strategy is selected."
            " Evaluation will be run against all project documents"
        )
