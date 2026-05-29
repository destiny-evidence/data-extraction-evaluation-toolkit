import os
import shutil
import signal
import time
from pathlib import Path
from threading import Thread

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from deet.data_models.documents import Document
from deet.data_models.project import DeetProject
from deet.scripts.cli import app
from deet.settings import get_settings

settings = get_settings()


@pytest.fixture
def runner():
    return CliRunner()


INTEGRATION_DATASETS = [Path(__file__).parent / "datasets/ebmnlp_with_metadata"]


@pytest.fixture(scope="module")
def tmp_project_workspace(tmp_path_factory):
    """Create a workspace directory that persists for this module."""
    workspace_dir = tmp_path_factory.mktemp("projects")
    previous_cwd = Path.cwd()
    os.chdir(workspace_dir)
    try:
        yield workspace_dir
    finally:
        os.chdir(previous_cwd)


@pytest.fixture
def virtual_keyboard():
    with (
        create_pipe_input() as pipe_input,
        create_app_session(input=pipe_input, output=DummyOutput()),
    ):
        yield pipe_input


@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_initialise_project_via_wizard(
    runner, dataset_base_path, tmp_project_workspace, virtual_keyboard
):
    """Test whether Alice can initialise a project with her data using the wizard."""
    # Alice's project name is the last bit of the path to her dataset
    project_name = dataset_base_path.name

    # Alice will use a directory named after her project name in the
    # tmp_project_workspace created for testing deet
    project_dir = tmp_project_workspace / project_name

    # She checks if it exists, and removes it if it does, then creates it afresh
    # and changed into it
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    os.chdir(project_dir)

    def user_journey():
        """Replicate the keyboard strokes Alice will make when using the CLI."""
        # Alice sees an informative splash screen informing her on how to use the wizard
        # She presses enter to continue
        virtual_keyboard.send_text("\n")
        # Alice enters the name of her project name,
        virtual_keyboard.send_text(f"{project_name}\n")
        # The path to her dataset
        virtual_keyboard.send_text(f"{dataset_base_path / 'reports.json'}\n")
        # She selects the default dataset type option (eppijson)
        time.sleep(0.1)
        virtual_keyboard.send_text("\r")
        # She enters the path to her pdfs
        virtual_keyboard.send_text(f"{dataset_base_path / 'pdfs'}\n")
        virtual_keyboard.send_text("\n")
        virtual_keyboard.send_text(f"{settings.azure_api_key}\r")
        virtual_keyboard.send_text(f"{settings.azure_api_key}\r")

    typer_thread = Thread(target=user_journey, daemon=True)
    typer_thread.start()

    time_limit = 5

    def timeout_handler(signum, frame):
        out_of_time = f"The CLI wizard timed out after {time_limit} seconds!"
        raise TimeoutError(out_of_time)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(time_limit)

    result = runner.invoke(app, ["project", "init"])
    signal.alarm(0)

    assert result.exit_code == 0
    deet_project = DeetProject.load()
    assert deet_project.name == project_name

    processed_data = deet_project.process_data()
    assert len(processed_data.documents) > 0
    assert len(processed_data.attributes) > 0
    assert len(processed_data.annotated_documents) > 0


@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_linking_with_map(
    runner,
    dataset_base_path,
    tmp_project_workspace,
):
    """Test whether Alice can link documents."""
    # Alice makes sure she is in the project directory she created on project init
    project_dir = tmp_project_workspace / dataset_base_path.name
    os.chdir(project_dir)
    #  Alice adds the necessary metadata for her files to be linked
    shutil.copy(dataset_base_path / "link_map.csv", project_dir / "link_map.csv")

    result = runner.invoke(app, ["project", "link"])
    assert result.exit_code == 0

    deet_project = DeetProject.load()
    processed_data = deet_project.process_data()

    for doc in processed_data.documents:
        doc_id = doc.safe_identity.document_id
        doc_path = deet_project.linked_documents_path / f"{doc_id}.json"
        assert doc_path.exists()
        linked_doc = Document.load(doc_path)
        assert linked_doc.parsed_document is not None
        assert linked_doc.parsed_document.text != ""
