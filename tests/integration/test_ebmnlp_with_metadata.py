import shutil
from pathlib import Path

import pytest

from deet.scripts.cli import (
    init_linkage_mapping_file,
    init_prompt_csv,
    link_documents_fulltexts,
)


@pytest.mark.parametrize(
    "dataset_base_path", [Path(__file__).parent / "datasets/ebmnlp_with_metadata"]
)
@pytest.mark.xfail
def test_can_run_toolkit(tmp_path, dataset_base_path):
    # Alice has some data she wants extracted from some PDFs
    # She already has markdown versions of her PDFs and she has
    # an eppi.json file with some gold standard data and annotations she
    # wants extracted.

    # She starts by creating a new folder for this project
    project_folder = tmp_path
    # and adds her gold standard eppi.json to it
    shutil.copy(dataset_base_path / "reports.json", tmp_path)

    # Following the documentation instructions Alice starts by generating
    # a linkage mapping file
    gs_data_path = project_folder / "reports.json"
    link_map_path = project_folder / "link_map.csv"
    init_linkage_mapping_file(
        gs_data_path=gs_data_path,
        link_map_path=link_map_path,
    )

    assert link_map_path.exists()

    #  Alice adds the necessary metadata for her files to be linked
    shutil.copy(dataset_base_path / "link_map.csv", link_map_path)

    # Alice adds her markdown files of her PDFs to a project subfolder
    documents_path = project_folder / "pdfs"
    shutil.copytree(dataset_base_path / "pdfs", documents_path)

    original_documents = list(documents_path.iterdir())
    assert len(original_documents) > 0
    linked_documents_path = project_folder / "linked_documents"

    # Alice links here documents before being able to extract data from them
    link_documents_fulltexts(
        gs_data_path=gs_data_path,
        link_map_path=link_map_path,
        pdf_dir=documents_path,
        output_path=linked_documents_path,
    )

    assert linked_documents_path.exists()
    created_linked_files = list(linked_documents_path.iterdir())
    assert len(created_linked_files) == len(original_documents)
    assert all(file.suffix == ".json" for file in created_linked_files)

    # Alice then generates her prompt.csv so she can write her prompts before
    # extraction
    prompt_csv_path = project_folder / "prompt_definitions.csv"
    init_prompt_csv(
        gs_data_path=gs_data_path,
        csv_path=prompt_csv_path,
    )

    assert prompt_csv_path.exists()

    # Alice populates the default prompt template csv with her prompts
    shutil.copy(dataset_base_path / "prompt_definitions.csv", prompt_csv_path)

    # Alice is now ready to extract
