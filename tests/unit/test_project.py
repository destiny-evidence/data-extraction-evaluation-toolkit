"""Tests for data_models/project.py."""

from unittest.mock import patch

from deet.data_models.project import DeetProject, ExperimentArtefacts


def test_deet_project_creates_artefacts(tmp_path, valid_project_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = DeetProject(**valid_project_data)

    project.setup()

    resource_paths = [
        project.experiments_dir,
        project.prompt_csv_path,
        project.link_map_path,
        project.linked_documents_path,
        project.config_path,
    ]

    for path in resource_paths:
        assert path.exists()


def test_init_extraction_run(tmp_path):
    """Ensure it creates the folder; ensure it creates deet.log."""
    out_dir = tmp_path / "experiments"
    out_dir.mkdir()
    run_name = "test_run"

    with patch("deet.data_models.project.logger") as mock_logger:
        experiment_artefacts = ExperimentArtefacts.create(out_dir, run_name)

    # run ID format contains timestamp and run name
    assert run_name in experiment_artefacts.run_id
    assert "_" in experiment_artefacts.run_id  # timestamp separator

    # check experiment directory was created
    assert experiment_artefacts.base_dir.exists()
    assert experiment_artefacts.base_dir.is_dir()
    assert experiment_artefacts.base_dir.parent == out_dir

    # check logger.add was called with log file path
    mock_logger.add.assert_called_once()
    log_path = mock_logger.add.call_args[0][0]
    assert log_path == experiment_artefacts.base_dir / "deet.log"
