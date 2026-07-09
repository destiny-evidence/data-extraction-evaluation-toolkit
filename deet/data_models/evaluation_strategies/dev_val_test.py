# ruff: noqa: PLC0415
"""
The dynamic dev-val-test splitting strategy.

This is described in detail at https://destiny-evidence.github.io/evaluation-book/index-1/#chunked-evaluation-data
"""

from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from deet.data_models.enums import EvaluationStrategyName
from deet.data_models.evaluation_strategies.base import (
    BaseEvaluationStage,
    BaseEvaluationStrategy,
    BaseSplits,
)
from deet.data_models.project import DeetProject
from deet.exceptions import SplitsValidationError
from deet.ui import fail_with_message, notify

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

    from deet.data_models.project import DeetProject, ExperimentArtefacts


class DevValTestEvaluationStage(BaseEvaluationStage):
    """
    Describes the possible evaluation stages.

    DEVELOPMENT is used to iterate and improve prompts/configuration.
    VALIDATION is used to validate prompts on data they have not been tuned for.
    TEST is used for a final assessment.
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
    def load(cls, file_path: Path) -> DevValTestSplits:
        """Load splits from file."""
        if not file_path.exists():
            return cls()
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

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


class DevValTestEvaluationStrategy(BaseEvaluationStrategy[DevValTestSplits]):
    """Strategy to manage dynamic splitting into dev-val-test."""

    name = EvaluationStrategyName.DEV_VAL_TEST

    def _load_splits(self, project: DeetProject) -> DevValTestSplits:
        """Make a new splits object with all project IDs."""
        return DevValTestSplits.load(project.evaluation_splits_path)

    def _add_dev(
        self, size: int, project: DeetProject, project_doc_ids: list[int]
    ) -> None:
        """Add unassigned documents to the development pool."""
        try:
            n_added = self.splits.add_to_stage(
                DevValTestEvaluationStage.DEVELOPMENT, project_doc_ids, size
            )
            self.splits.dump_to_json(project.evaluation_splits_path)

        except SplitsValidationError as e:
            fail_with_message(str(e))

        notify(
            f"Added {n_added} documents to development set."
            f" This now contains {len(self.splits.development_ids)} documents."
            f" {len(self.splits.get_unassigned_ids(project_doc_ids))}"
            " are still unassigned."
        )

    def _validate_run(
        self, deet_project: DeetProject, size: int, project_doc_ids: list[int]
    ) -> None:
        """Select a past experiment config and eval against a fresh validation set."""
        from InquirerPy import inquirer

        from deet.data_models.project import ExperimentArtefacts
        from deet.extractors.cli_helpers import (
            evaluate_extraction_pipeline,
            run_extraction_pipeline,
        )
        from deet.ui import fail_with_message, notify

        all_experiments = [
            ExperimentArtefacts(base_dir=path)
            for path in deet_project.experiments_dir.iterdir()
            if path.is_dir()
        ]
        completed_experiments = [exp for exp in all_experiments if exp.is_complete]
        completed_experiments.sort(key=lambda e: e.run_id, reverse=True)

        choices = [{"name": exp.run_id, "value": exp} for exp in completed_experiments]

        selected_experiment: ExperimentArtefacts = inquirer.select(
            message="Select the experiment configuration to validate:", choices=choices
        ).execute()

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
            f"\nEvaluating experiment: {selected_experiment.run_id}"
            " using these documents"
        )

        run_output, processed_annotation_data, experiment_artefacts = (
            run_extraction_pipeline(
                deet_project=deet_project,
                config_path=selected_experiment.config_snapshot,
                run_name="VALIDATION",
            )
        )
        evaluate_extraction_pipeline(
            processed_annotation_data=processed_annotation_data,
            run_output=run_output,
            experiment_artefacts=experiment_artefacts,
        )
        self.splits.validation_run_id = experiment_artefacts.run_id
        self.splits.dump_to_json(deet_project.evaluation_splits_path)
        self.snapshot(experiment_artefacts)

    def _act_on_validation(
        self, deet_project: DeetProject, project_doc_ids: list[int]
    ) -> None:
        """Given validation, choose to return to development or move to testing."""
        from InquirerPy import inquirer

        from deet.data_models.project import ExperimentArtefacts
        from deet.extractors.cli_helpers import (
            evaluate_extraction_pipeline,
            run_extraction_pipeline,
        )
        from deet.ui import fail_with_message

        decision = inquirer.select(
            message="Based on these metrics, how would you like to proceed?",
            choices=[
                {
                    "name": (
                        "Accept: lock this configuration, and "
                        " do a final test on all remaining documents."
                    ),
                    "value": "accept",
                },
                {
                    "name": (
                        "Reject: add validation documents to "
                        " development set and continue iterating."
                    ),
                    "value": "reject",
                },
            ],
        ).execute()

        if decision == "accept":
            try:
                self.splits.finalise_test(project_doc_ids)
            except SplitsValidationError as e:
                fail_with_message(str(e))

            if self.splits.validation_run_id is None:
                fail_with_message("No validation run id")
            selected_experiment = ExperimentArtefacts(
                base_dir=deet_project.experiments_dir / self.splits.validation_run_id
            )

            self.splits.dump_to_json(deet_project.evaluation_splits_path)

            run_output, processed_annotation_data, experiment_artefacts = (
                run_extraction_pipeline(
                    deet_project=deet_project,
                    config_path=selected_experiment.config_snapshot,
                    run_name="FINAL_TEST",
                )
            )
            evaluate_extraction_pipeline(
                processed_annotation_data=processed_annotation_data,
                run_output=run_output,
                experiment_artefacts=experiment_artefacts,
            )

        elif decision == "reject":
            self.splits.reject_validation()
            self.splits.dump_to_json(deet_project.evaluation_splits_path)

    def run_splits_wizard(
        self,
        project: DeetProject,
        *,
        action: str | None = None,
        size: int | None = None,
        experiment: str | None = None,
    ) -> None:
        """Run the splits wizard."""
        from InquirerPy import inquirer

        project_doc_ids = project.get_all_doc_ids()
        unassigned = self.splits.get_unassigned_ids(project_doc_ids)

        notify(
            f"Strategy:    dev/val/test\n"
            f"Stage:       {self.splits.current_stage}\n"
            f"Development: {len(self.splits.development_ids)} documents\n"
            f"Validation:  {len(self.splits.validation_ids)} documents\n"
            f"Test:        {len(self.splits.test_ids)} documents\n"
            f"Unassigned:  {len(unassigned)} documents"
        )

        if self.splits.current_stage == DevValTestEvaluationStage.DEVELOPMENT:
            choices = [{"name": "Add documents to development set", "value": "add-dev"}]
            if self.splits.development_ids:
                choices.append({"name": "Move to validation", "value": "validate"})

            if action is None:
                action = inquirer.select(
                    message="what would you like to do?", choices=choices
                ).execute()

            if action == "add-dev":
                if size is None:
                    size = int(
                        inquirer.number(
                            message="How many documents would you like to add?"
                        ).execute()
                    )
                self._add_dev(size, project, project_doc_ids)
                return

            if action == "validate":
                if size is None:
                    size = int(
                        inquirer.number(
                            message="How many documents would you like to add?"
                        ).execute()
                    )
                self._validate_run(
                    deet_project=project, size=size, project_doc_ids=project_doc_ids
                )

        if self.splits.current_stage == DevValTestEvaluationStage.VALIDATION:
            self._act_on_validation(
                deet_project=project, project_doc_ids=project_doc_ids
            )

        elif self.splits.current_stage == DevValTestEvaluationStage.TEST:
            notify("Test is complete. No further action available")
