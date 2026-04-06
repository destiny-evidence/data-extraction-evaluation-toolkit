"""Tests for data_models/project.py."""

import json

import pytest

from deet.data_models.project import DeetProject
from deet.processors.converter_register import SupportedImportFormat


@pytest.fixture
def valid_project_data(tmp_path, sample_eppi_data):
    # Create a real dummy file so Pydantic's FilePath is happy
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(sample_eppi_data))

    return {
        "name": "TestProject",
        "gold_standard_data_path": data_file,
        "gold_standard_data_format": SupportedImportFormat.EPPI_JSON,
        "environment_file": "project",
        "pdf_dir": tmp_path,  # A real directory
    }


def test_deet_project_creates_artifacts(tmp_path, valid_project_data, monkeypatch):
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
