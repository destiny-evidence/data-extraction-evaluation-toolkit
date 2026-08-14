"""Tests for deet/data_models/evaluation_strategies/."""

import json
from unittest.mock import MagicMock, patch

import pytest

from deet.data_models.enums import EvaluationStrategyName
from deet.data_models.evaluation_strategies import _STRATEGY_REGISTRY
from deet.data_models.evaluation_strategies.dev_val_test import (
    DevValTestEvaluationStage,
    DevValTestEvaluationStrategy,
    DevValTestSplits,
)
from deet.data_models.evaluation_strategies.null import NullEvaluationStrategy


def test_strategy_registry_covers_all_strategy_names():
    assert set(_STRATEGY_REGISTRY.keys()) == set(EvaluationStrategyName)


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

    assert DevValTestSplits.load_or_init(path) == splits


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

    with patch("InquirerPy.inquirer") as mock_inquirer:
        mock_inquirer.select.return_value.execute.return_value = "reject"
        strategy._act_on_validation(
            deet_project=mock_project, project_doc_ids=[1, 2, 3, 4]
        )

    reloaded = DevValTestSplits.load_or_init(splits_path)
    assert reloaded.current_stage == DevValTestEvaluationStage.DEVELOPMENT
    assert set(reloaded.development_ids) == {1, 2, 3, 4}
    assert reloaded.validation_ids == []


def test_add_dev_samples_and_persists(tmp_path):
    splits = DevValTestSplits()
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path

    strategy = DevValTestEvaluationStrategy(mock_project)
    strategy.add_to_development(
        size=2, project=mock_project, project_doc_ids=[1, 2, 3, 4, 5]
    )

    reloaded = DevValTestSplits.load_or_init(splits_path)
    assert len(reloaded.development_ids) == 2
    assert all(d in [1, 2, 3, 4, 5] for d in reloaded.development_ids)


def test_act_on_validation_accept_fails_without_validation_run_id(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.VALIDATION,
        development_ids=[1, 2],
        validation_ids=[3, 4],
        validation_run_id=None,
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path

    strategy = DevValTestEvaluationStrategy(mock_project)

    with (
        patch("InquirerPy.inquirer") as mock_inquirer,
        patch("deet.ui.fail_with_message", side_effect=SystemExit) as mock_fail,
    ):
        mock_inquirer.select.return_value.execute.return_value = "accept"
        with pytest.raises(SystemExit):
            strategy._act_on_validation(
                deet_project=mock_project, project_doc_ids=[1, 2, 3, 4, 5]
            )

    assert "validation run" in mock_fail.call_args[0][0].lower()


def test_act_on_validation_accept_finalises_test_and_runs_pipeline(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.VALIDATION,
        development_ids=[1, 2],
        validation_ids=[3, 4],
        validation_run_id="2024-01-01_run",
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    exp_dir = tmp_path / "experiments" / "2024-01-01_run"
    exp_dir.mkdir(parents=True)
    config_snapshot = exp_dir / "config.yaml"
    config_snapshot.write_text("")

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path
    mock_project.experiments_dir = tmp_path / "experiments"

    strategy = DevValTestEvaluationStrategy(mock_project)

    with (
        patch("InquirerPy.inquirer") as mock_inquirer,
        patch("deet.extractors.cli_helpers.run_extraction_pipeline") as mock_run,
        patch("deet.extractors.cli_helpers.evaluate_extraction_pipeline"),
    ):
        mock_inquirer.select.return_value.execute.return_value = "accept"
        mock_run.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
        strategy._act_on_validation(
            deet_project=mock_project, project_doc_ids=[1, 2, 3, 4, 5]
        )

    reloaded = DevValTestSplits.load_or_init(splits_path)
    assert reloaded.current_stage == DevValTestEvaluationStage.TEST
    assert reloaded.test_ids == [5]
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["config_path"] == config_snapshot


def test_run_splits_wizard_test_stage_notifies_and_exits(tmp_path):
    splits = DevValTestSplits(
        current_stage=DevValTestEvaluationStage.TEST,
        test_ids=[1, 2, 3],
    )
    splits_path = tmp_path / "splits.json"
    splits.dump_to_json(splits_path)

    mock_project = MagicMock()
    mock_project.evaluation_splits_path = splits_path
    mock_project.get_all_doc_ids.return_value = [1, 2, 3]

    strategy = DevValTestEvaluationStrategy(mock_project)

    with (
        patch.object(strategy, "_act_on_validation") as mock_act,
        patch(
            "deet.data_models.evaluation_strategies.dev_val_test.notify"
        ) as mock_notify,
    ):
        strategy.run_splits_wizard(project=mock_project)

    mock_act.assert_not_called()
    assert mock_notify.call_count == 2  # status summary + completion message


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
