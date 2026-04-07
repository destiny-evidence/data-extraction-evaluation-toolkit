"""Tests for data_models/project.py."""

from deet.data_models.project import DeetProject


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
