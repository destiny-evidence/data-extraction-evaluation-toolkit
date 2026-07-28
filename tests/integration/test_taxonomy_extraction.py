import os
import shutil
from pathlib import Path

import pytest

from deet.data_models.project import DeetProject
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
        vocabulary_path=dataset_base_path / "taxonomy.ttl",
        vocabulary_mapping_path=dataset_base_path / "taxonomy_nacsos_mapping.json",
    )
    project.setup()

    try:
        yield project_dir
    finally:
        os.chdir(previous_cwd)


@pytest.fixture
def dataset_base_path():
    return Path(__file__).parent / "datasets" / "climate_health_taxonomy"


def test_project_loads_vocabulary_and_links(initialised_project_workspace):
    project = DeetProject.load()
    schemes = project.load_schemes()
    assert len(schemes) == 22
    total = sum(len(scheme.concepts) for scheme in schemes)
    assert total == 496

    processed_data = project.process_data()
    attributes = processed_data.attributes

    linked_schemes = [
        scheme.map_concepts(project.vocabulary_mapping_path, attributes)
        for scheme in schemes
    ]
    assert len(linked_schemes) == len(schemes)

    linked_total = sum(len(scheme.concepts) for scheme in linked_schemes)
    attributes_to_link = [att for att in attributes if "|" in att.attribute_label]
    assert linked_total == len(attributes_to_link), "Not all attributes could be linked"
