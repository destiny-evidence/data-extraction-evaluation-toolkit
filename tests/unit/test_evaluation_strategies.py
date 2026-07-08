"""Tests for deet/data_models/evaluation_strategies/."""

import json
from unittest.mock import MagicMock, patch

from deet.data_models.evaluation_strategies.dev_val_test import (
    DevValTestEvaluationStage,
    DevValTestEvaluationStrategy,
    DevValTestSplits,
)
from deet.data_models.evaluation_strategies.null import NullEvaluationStrategy


def test_null_strategy_get_active_ids_returns_all_doc_ids():
    mock_project = MagicMock()
    mock_project.get_all_doc_ids.return_value = [1, 2, 3]

    strategy = NullEvaluationStrategy(mock_project)

    assert strategy.get_active_ids(mock_project) == [1, 2, 3]


def test_null_strategy_snapshot_writes_json(tmp_path):
    mock_project = MagicMock()
    mock_project.get_all_doc_ids.return_value = [1, 2, 3]
    mock_artefacts = MagicMock()
    mock_artefacts.evaluation_splits_snapshot = tmp_path / "evaluation_splits.json"

    strategy = NullEvaluationStrategy(mock_project)
    strategy.snapshot(mock_artefacts)

    data = json.loads(mock_artefacts.evaluation_splits_snapshot.read_text())
    assert data["all_ids"] == [1, 2, 3]


def test_dev_val_test_splits_round_trips(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.DEVELOPMENT,
        development_ids=[1, 2, 3],
        validation_ids=[4, 5],
        test_ids=[],
        validation_run_id="val_run",
    )
    path = tmp_path / "splits.json"
    splits.dump_to_json(path)

    assert DevValTestSplits.load(path) == splits


def test_run_splits_wizard_add_dev_dispatches_correctly(tmp_path):
    splits = DevValTestSplits(development_ids=[1, 2])
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path
    mock_project.get_all_doc_ids.return_value = [1, 2, 3, 4, 5]

    strategy = DevValTestEvaluationStrategy(mock_project)

    with (
        patch.object(strategy, "_add_dev") as mock_add_dev,
        patch.object(strategy, "_validate_run") as mock_validate_run,
    ):
        strategy.run_splits_wizard(
            project=mock_project,
            typer_context=MagicMock(),
            action="add-dev",
            size=2,
        )

    mock_add_dev.assert_called_once_with(2, mock_project, [1, 2, 3, 4, 5])
    mock_validate_run.assert_not_called()


def test_run_splits_wizard_validation_stage_dispatches_to_act_on_validation(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.VALIDATION,
        development_ids=[1, 2],
        validation_ids=[3, 4],
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path
    mock_project.get_all_doc_ids.return_value = [1, 2, 3, 4, 5]

    strategy = DevValTestEvaluationStrategy(mock_project)

    with patch.object(strategy, "_act_on_validation") as mock_act:
        strategy.run_splits_wizard(
            project=mock_project,
            typer_context=MagicMock(),
            action="accept",
        )

    mock_act.assert_called_once()


def test_act_on_validation_reject_returns_to_development(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.VALIDATION,
        development_ids=[1, 2],
        validation_ids=[3, 4],
        validation_run_id="some-run",
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path
    mock_project.get_all_doc_ids.return_value = [1, 2, 3, 4]

    strategy = DevValTestEvaluationStrategy(mock_project)

    mock_ctx = MagicMock()
    mock_ctx.obj.project = mock_project

    with patch(
        "deet.data_models.evaluation_strategies.dev_val_test.inquirer"
    ) as mock_inquirer:
        mock_inquirer.select.return_value.execute.return_value = "reject"
        strategy._act_on_validation(
            typer_context=mock_ctx, project_doc_ids=[1, 2, 3, 4]
        )

    reloaded = DevValTestSplits.load(splits_path)
    assert reloaded.current_stage == DevValTestEvaluationStage.DEVELOPMENT
    assert set(reloaded.development_ids) == {1, 2, 3, 4}
    assert reloaded.validation_ids == []


def test_dev_val_test_strategy_get_active_ids_returns_current_stage_ids(tmp_path):
    splits = DevValTestSplits(
        development_ids=[1, 2],
        validation_ids=[3, 4],
        test_ids=[5],
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path

    strategy = DevValTestEvaluationStrategy(mock_project)

    assert strategy.get_active_ids(mock_project) == [1, 2]
