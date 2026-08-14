# ruff: noqa: PLC0415
"""
The dynamic dev-val-test splitting strategy.

This is described in detail at https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data
"""

from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, ClassVar, TypedDict

from loguru import logger
from pydantic import Field, ValidationError

from deet.data_models.enums import EvaluationStrategyName
from deet.data_models.evaluation_strategies.base import (
    BaseEvaluationStage,
    BaseEvaluationStrategy,
    BaseSplits,
)
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.exceptions import SplitsValidationError
from deet.ui import fail_with_message, notify

if TYPE_CHECKING:
    from collections.abc import Callable, Collection
    from pathlib import Path


class DevValTestEvaluationStage(BaseEvaluationStage):
    """
    Describes the possible evaluation stages.

    Inherits from BaseEvaluationStage, and defines each of the following stages
    for the dev-val-test strategy.

    - DEVELOPMENT is used to iterate and improve prompts/configuration.
    - VALIDATION is used to validate prompts on data they have not been tuned for.
    - TEST is used for a final assessment.
    """

    DEVELOPMENT = auto()
    VALIDATION = auto()
    TEST = auto()


class DevValTestSplits(BaseSplits):
    """Model to record how documents are allocated across dev-val-test splits."""

    _STAGE_FIELD_NAMES: ClassVar[dict[BaseEvaluationStage, str]] = {
        DevValTestEvaluationStage.DEVELOPMENT: "development_ids",
        DevValTestEvaluationStage.VALIDATION: "validation_ids",
        DevValTestEvaluationStage.TEST: "test_ids",
    }

    current_stage: DevValTestEvaluationStage = DevValTestEvaluationStage.DEVELOPMENT
    development_ids: list[int] = Field(default_factory=list)
    validation_ids: list[int] = Field(default_factory=list)
    test_ids: list[int] = Field(default_factory=list)

    validation_run_id: str | None = None

    @classmethod
    def load_or_init(cls, file_path: Path) -> DevValTestSplits:
        """
        Load splits from file, or initialise if empty or invalid.

        Note:
            It may be invalid if the evaluation strategy was switched.
            In this case, it would make sense to start a fresh instantiation

        """
        if not file_path.exists():
            logger.warning(
                "No splits file exists at {file_path}. Instantiating fresh instance"
            )
            return cls()
        try:
            return cls.model_validate_json(file_path.read_text(encoding="utf-8"))
        except ValidationError:
            logger.warning(
                "Existing splits file does not match this strategy's schema"
                " Instantiating fresh splits."
            )
            return cls()

    def finalise_test(self, project_doc_ids: Collection[int]) -> None:
        """Add all remaining docs to test."""
        unassigned = self.get_unassigned_ids(project_doc_ids)

        if len(unassigned) == 0:
            none_remaining = (
                "No unassigned documents left for testing."
                " Add more documents to the project to continue."
            )
            raise SplitsValidationError(none_remaining)

        self.test_ids = unassigned
        self.current_stage = DevValTestEvaluationStage.TEST

    def reject_validation(self) -> None:
        """Merge validation IDs into development and continue developing."""
        self.development_ids.extend(self.validation_ids)
        self.validation_ids = []
        self.current_stage = DevValTestEvaluationStage.DEVELOPMENT


class EvaluationDecisionSpec(TypedDict):
    """Definition of the shape of choices presented to user, and how to act on them."""

    name: str
    description: str
    execute: Callable[
        [DevValTestEvaluationStrategy, list[int], int | None, str | None], None
    ]


STAGE_ACTIONS: dict[DevValTestEvaluationStage, list[EvaluationDecisionSpec]] = {
    DevValTestEvaluationStage.DEVELOPMENT: [
        {
            "name": "add-dev",
            "description": "Add documents to development set",
            "execute": lambda strategy,
            project_ids,
            size,
            _experiment: strategy.add_to_development(project_ids, size),
        }
    ],
    DevValTestEvaluationStage.VALIDATION: [
        {
            "name": "accept",
            "description": (
                "Accept: lock this configuration, and do a"
                " final test on all remaining documents"
            ),
            "execute": lambda strategy,
            project_ids,
            _size,
            _experiment: strategy.accept_validation(project_ids),
        },
        {
            "name": "reject",
            "description": (
                "Reject: add validation documents to "
                " development set and continue iterating."
            ),
            "execute": lambda strategy,
            _project_ids,
            _size,
            _experiment: strategy.splits.reject_validation(),
        },
    ],
}

ADD_TO_DEVELOPMENT: EvaluationDecisionSpec = {
    "description": "Move to validation",
    "name": "validate",
    "execute": lambda strategy,
    project_ids,
    size,
    experiment: strategy.run_validation_interactive(
        project_ids, size=size, experiment=experiment
    ),
}


class DevValTestEvaluationStrategy(BaseEvaluationStrategy[DevValTestSplits]):
    """Strategy to manage dynamic splitting into dev-val-test."""

    name = EvaluationStrategyName.DEV_VAL_TEST

    def _load_splits(self, project: DeetProject) -> DevValTestSplits:
        """Make a new splits object with all project IDs."""
        return DevValTestSplits.load_or_init(project.evaluation_splits_path)

    def add_to_development(
        self,
        project_doc_ids: list[int],
        size: int | None = None,
    ) -> None:
        """
        Add unassigned documents to the development pool.

        Randomly samples from the unassigns documents and adds them
        to the development set.

        Persists the updated splits to dist and notifies the user of actions taken.

        Args:
            project_doc_ids: All document IDs in the project
                (to determine which are unassigned)
            size: Number of documents to randomly sample and add (prompted for if None).

        Raises:
            SplitsValidationError: If size exceeds the number of unassigned documents

        """
        from InquirerPy import inquirer

        if not size:
            size = int(
                inquirer.number(
                    message="How many documents would you like to add?"
                ).execute()
            )
        try:
            n_added = self.splits.add_to_stage(
                DevValTestEvaluationStage.DEVELOPMENT, project_doc_ids, size
            )
            self.splits.dump_to_json(self._project.evaluation_splits_path)

        except SplitsValidationError as e:
            fail_with_message(str(e))

        notify(
            f"Added {n_added} documents to development set."
            f" This now contains {len(self.splits.development_ids)} documents."
            f" {len(self.splits.get_unassigned_ids(project_doc_ids))}"
            " are still unassigned."
        )

    def validate_run(
        self, size: int, project_doc_ids: list[int], experiment: ExperimentArtefacts
    ) -> None:
        """
        Evaluate a past experiment config and eval against a fresh validation set.

        Randomly samples previously unsassigned documents into the validation set,
        evaluates given experiment config and prompts against them,
        and updates the splits state with the validation run ID for later reference

        Persists the updated splits and a snapshot of strategy state with the
        experiment artefacts.

        Args:
            size: Number of documents to randomly sample
            project_doc_ids: All document IDs in the project (to determine unassigned)
            experiment: The ExperimentArtefacts to evaluate (from a prior run)

        Raises:
            SplitsValidationError: If size exceeds the number of unassigned documents

        Note:
            The CLI calls `choose_experiment()` interactively, then passes it here.
            For programmatic use, provide the experiment directly.

        """
        from deet.extractors.cli_helpers import (
            evaluate_extraction_pipeline,
            run_extraction_pipeline,
        )
        from deet.ui import fail_with_message, notify

        try:
            n_added = self.splits.add_to_stage(
                DevValTestEvaluationStage.VALIDATION, project_doc_ids, size
            )
            self.splits.current_stage = DevValTestEvaluationStage.VALIDATION
        except SplitsValidationError as e:
            fail_with_message(str(e))

        notify(
            f"Added {n_added} documents to validation set"
            f" ({len(self.splits.get_unassigned_ids(project_doc_ids))}"
            " are still unassigned)."
            f"\nEvaluating experiment: {experiment.run_id}"
            " using these documents"
        )

        run_output, processed_annotation_data, experiment_artefacts, _config = (
            run_extraction_pipeline(
                deet_project=self._project,
                prompt_csv_path=experiment.prompts_snapshot,
                config_path=experiment.config_snapshot,
                run_name="VALIDATION",
            )
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed_annotation_data,
            run_output=run_output,
            experiment_artefacts=experiment_artefacts,
        )
        self.splits.validation_run_id = experiment_artefacts.run_id
        self.splits.dump_to_json(self._project.evaluation_splits_path)
        self.snapshot(experiment_artefacts)

    def run_validation_interactive(
        self, project_doc_ids: list[int], size: int | None, experiment: str | None
    ) -> None:
        """Orchestrate validation (prompt for args and dispatch validation)."""
        from InquirerPy import inquirer

        from deet.ui.terminal.prompts import select_experiment

        if size is None:
            size = int(
                inquirer.number(
                    message="How many documents would you like to add?"
                ).execute()
            )
        if experiment is None:
            selected_experiment = select_experiment(self._project)
        else:
            selected_experiment = ExperimentArtefacts(
                base_dir=self._project.experiments_dir / experiment
            )
        self.validate_run(
            size=size, project_doc_ids=project_doc_ids, experiment=selected_experiment
        )

    def accept_validation(self, project_doc_ids: list[int]) -> None:
        """Accept the results of validation run and do final evaluation of test set."""
        from deet.data_models.project import ExperimentArtefacts
        from deet.extractors.cli_helpers import (
            evaluate_extraction_pipeline,
            run_extraction_pipeline,
        )
        from deet.ui import fail_with_message

        try:
            self.splits.finalise_test(project_doc_ids)
        except SplitsValidationError as e:
            fail_with_message(str(e))

        if self.splits.validation_run_id is None:
            fail_with_message("No validation run id")
        selected_experiment = ExperimentArtefacts(
            base_dir=self._project.experiments_dir / self.splits.validation_run_id
        )

        self.splits.dump_to_json(self._project.evaluation_splits_path)

        run_output, processed_annotation_data, experiment_artefacts, _config = (
            run_extraction_pipeline(
                deet_project=self._project,
                prompt_csv_path=selected_experiment.prompts_snapshot,
                config_path=selected_experiment.config_snapshot,
                run_name="FINAL_TEST",
            )
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed_annotation_data,
            run_output=run_output,
            experiment_artefacts=experiment_artefacts,
        )

    def run_splits_wizard(
        self,
        project: DeetProject,
        *,
        action: str | None = None,
        size: int | None = None,
        experiment: str | None = None,
    ) -> None:
        """Run the splits wizard."""
        from deet.ui.terminal.prompts import select_from_list

        project_doc_ids = project.get_all_doc_ids()
        unassigned = self.splits.get_unassigned_ids(project_doc_ids)

        notify(
            f"Strategy:    {self.name}\n"
            f"Stage:       {self.splits.current_stage}\n"
            f"Development: {len(self.splits.development_ids)} documents\n"
            f"Validation:  {len(self.splits.validation_ids)} documents\n"
            f"Test:        {len(self.splits.test_ids)} documents\n"
            f"Unassigned:  {len(unassigned)} documents"
        )

        if self.splits.current_stage == DevValTestEvaluationStage.DEVELOPMENT:
            choices = list(STAGE_ACTIONS[DevValTestEvaluationStage.DEVELOPMENT])
            if self.splits.development_ids:
                choices.append(ADD_TO_DEVELOPMENT)

            selected_action = select_from_list(choices, item_key="name")
            selected_action["execute"](self, project_doc_ids, size, experiment)

        if self.splits.current_stage == DevValTestEvaluationStage.VALIDATION:
            choices = STAGE_ACTIONS[DevValTestEvaluationStage.VALIDATION]
            selected_action = select_from_list(choices, item_key="name")
            selected_action["execute"](self, project_doc_ids, size, experiment)

        elif self.splits.current_stage == DevValTestEvaluationStage.TEST:
            notify("Test is complete. No further action available")

        self.splits.dump_to_json(self._project.evaluation_splits_path)
