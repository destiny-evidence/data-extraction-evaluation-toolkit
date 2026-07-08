"""Tests for data_models/project.py."""

import json
from unittest.mock import patch

import pytest

from deet.data_models.project import PROJECT_FILE, DeetProject, ExperimentArtefacts
from deet.processors.converter_register import SupportedImportFormat


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


def _make_project(root, data_file):
    """Build a project anchored to ``root`` pointing at ``data_file``."""
    project = DeetProject(
        name="anchored",
        gold_standard_data_path=data_file,
        gold_standard_data_format=SupportedImportFormat.EPPI_JSON,
    )
    project.anchor_to(root)
    return project


def test_anchor_to_stores_relative_paths_and_dumps_portable_yaml(
    tmp_path, sample_eppi_data
):
    # Data lives outside the project folder, as with `deet project new`.
    data_file = tmp_path / "data" / "reports.json"
    data_file.parent.mkdir()
    data_file.write_text(json.dumps(sample_eppi_data))
    project_root = tmp_path / "proj"

    project = _make_project(project_root, data_file)

    # Stored value is relative; the absolute helper still resolves correctly.
    assert not project.gold_standard_data_path.is_absolute()
    assert project.gold_standard_data_abspath.resolve() == data_file.resolve()

    project.setup()

    yaml_text = (project_root / PROJECT_FILE).read_text()
    assert str(tmp_path) not in yaml_text  # no absolute paths leaked
    assert "reports.json" in yaml_text


def test_load_reads_project_from_given_directory(tmp_path, sample_eppi_data):
    data_file = tmp_path / "data" / "reports.json"
    data_file.parent.mkdir()
    data_file.write_text(json.dumps(sample_eppi_data))
    project_root = tmp_path / "proj"

    _make_project(project_root, data_file).setup()

    loaded = DeetProject.load(project_root)

    assert loaded.root == project_root.resolve()
    assert loaded.gold_standard_data_abspath.resolve() == data_file.resolve()


def test_load_raises_when_no_project_in_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No project"):
        DeetProject.load(tmp_path)


def test_validate_resources_raises_for_missing_data(tmp_path):
    project = _make_project(tmp_path / "proj", tmp_path / "does-not-exist.json")

    with pytest.raises(FileNotFoundError, match="Gold standard data not found"):
        project.validate_resources()
