import os
import shutil
from pathlib import Path

import pytest

from deet.data_models.project import DeetProject
from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.processors.converter_register import SupportedImportFormat


@pytest.fixture
def initialised_project_workspace(tmp_project_workspace, dataset_base_path):
    """Set up an initialised project, on which other tests depend."""
    previous_cwd = Path.cwd()

    project_name = dataset_base_path.name
    project_dir = tmp_project_workspace / project_name

    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    os.chdir(project_dir)

    # Programmatically mock a completed wizard setup matching the dataset
    project = DeetProject(
        name=project_name,
        gold_standard_data_path=dataset_base_path / "taxonomy_resolved_clean.csv",
        gold_standard_data_format=SupportedImportFormat.GENERIC_CSV,
        pdf_dir=None,
    )
    project.setup()

    try:
        yield project_dir
    finally:
        os.chdir(previous_cwd)


@pytest.fixture
def dataset_base_path():
    return Path(__file__).parent / "datasets" / "climate_health_taxonomy"


@pytest.fixture
def top_down_config(dataset_base_path):
    return DataExtractionConfig(
        method=ExtractionMethod.HIERARCHICAL_TOP_DOWN,
        vocabulary_path=dataset_base_path / "taxonomy.ttl",
        vocabulary_mapping_path=dataset_base_path / "taxonomy_nacsos_mapping.json",
    )
