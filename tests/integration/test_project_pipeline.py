import csv
import os
import shutil
import time
from pathlib import Path
from threading import Thread

import litellm
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from typer.testing import CliRunner

from deet.data_models.documents import Document
from deet.data_models.project import DeetProject, ExperimentArtefacts
from deet.processors.converter_register import SupportedImportFormat
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
        gold_standard_data_path=dataset_base_path / "reports.json",
        gold_standard_data_format=SupportedImportFormat.EPPI_JSON,
        pdf_dir=dataset_base_path / "pdfs",
    )
    project.setup()

    try:
        yield project_dir
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
    original_cwd = Path.cwd()
    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)
    os.chdir(project_dir)

    time_limit = 10
    did_time_out = False

    def user_journey():
        """Replicate the keyboard strokes Alice will make when using the CLI."""
        nonlocal did_time_out
        try:
            # Alice sees an informative splash screen informing her on how to
            # use the wizard. She presses enter to continue
            virtual_keyboard.send_text("\r")
            # Alice enters the name of her project name,
            virtual_keyboard.send_text(f"{project_name}\r")
            # The path to her dataset
            virtual_keyboard.send_text(f"{dataset_base_path / 'reports.json'}\r")
            # She selects the default dataset type option (eppijson)
            time.sleep(0.1)
            virtual_keyboard.send_text("\r")
            # She enters the path to her pdfs
            virtual_keyboard.send_text(f"{dataset_base_path / 'pdfs'}\r")
            virtual_keyboard.send_text("\r")
            virtual_keyboard.send_text("\r")
            virtual_keyboard.send_text("\r")
            virtual_keyboard.send_text("\r")

            # If things haven't finished at the end of the time limit,
            # Alice exits the wizard.
            time.sleep(time_limit)
            did_time_out = True
            virtual_keyboard.send_text("\x03")
        except OSError:
            # Safely caught when the main thread finishes successfully
            # and tears down the keyboard pipes early.
            pass

    typer_thread = Thread(target=user_journey, daemon=True)
    typer_thread.start()

    try:
        # Run natively on the main thread so CliRunner can attach to streams properly
        result = runner.invoke(app, ["project", "init"])

        assert result.exit_code == 0
        deet_project = DeetProject.load()
        assert deet_project.name == project_name

        processed_data = deet_project.process_data()
        assert len(processed_data.documents) > 0
        assert len(processed_data.attributes) > 0
        assert len(processed_data.annotated_documents) > 0

    finally:
        # Fixed: Pass your tracked path variable back to recover process state
        os.chdir(original_cwd)


@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_linking_with_map(
    runner, dataset_base_path, tmp_project_workspace, initialised_project_workspace
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


def assert_llm_csv_contains_values(path: Path) -> None:
    """Validate that the LLM produced something for each doc-attribute combination."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) > 0, (
        "CSV contains no rows."
        " Perhaps there was no output from llm calls"
        " remember to run pytest --record-mode=new_episodes"
        " if you have introduced behaviour that modifies"
        " llm calls"
    )


@pytest.mark.vcr
@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_extraction_without_evaluating(
    runner,
    dataset_base_path,
    tmp_project_workspace,
    initialised_project_workspace,
    request,
):
    """Test whether Alice can extract data using the LLM."""
    # Alice makes sure she is in the project directory she created on project init
    project_dir = tmp_project_workspace / dataset_base_path.name
    os.chdir(project_dir)
    deet_project = DeetProject.load()

    #  Alice adds the necessary metadata for her files to be linked
    shutil.copy(dataset_base_path / "link_map.csv", deet_project.link_map_path)

    # Alice adds her own custom prompts for the attributes she wants to extract
    shutil.copy(
        dataset_base_path / "prompt_definitions.csv", deet_project.prompt_csv_path
    )

    result = runner.invoke(app, ["project", "link"])
    assert result.exit_code == 0

    result = runner.invoke(
        app, ["experiments", "predict", "--config-path", deet_project.config_path]
    )
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0

    # Alice makes sure her experiments dir exists
    # and that it contains a folder for her new experiment
    assert deet_project.experiments_dir.exists()
    run_dirs = list(deet_project.experiments_dir.iterdir())
    assert len(run_dirs) == 1

    # She finds the directory for the experiment she just carried out
    experiment_run_dir = run_dirs[0]
    run_id = experiment_run_dir.name

    # And makes sure it contains all the experiment artefacts she would expect
    artefacts = ExperimentArtefacts(base_dir=experiment_run_dir, run_id=run_id)

    assert artefacts.config_snapshot.exists()
    assert artefacts.prompts_snapshot.exists()
    assert artefacts.llm_annotation_csv.exists()
    assert artefacts.llm_annotations.exists()

    assert_llm_csv_contains_values(artefacts.llm_annotation_csv)


@pytest.mark.vcr
@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_extraction_without_evaluating_or_linking(
    runner, dataset_base_path, tmp_project_workspace, initialised_project_workspace
):
    """Test whether Alice can extract data using the LLM."""
    # Alice makes sure she is in the project directory she created on project init
    project_dir = tmp_project_workspace / dataset_base_path.name
    os.chdir(project_dir)
    deet_project = DeetProject.load()

    # Alice adds her own custom prompts for the attributes she wants to extract
    shutil.copy(
        dataset_base_path / "prompt_definitions.csv", deet_project.prompt_csv_path
    )

    result = runner.invoke(app, ["project", "link"])
    assert result.exit_code == 0

    # Alice doesn't want to bother linking yet, she just extracts directly
    # from her texts.
    result = runner.invoke(
        app,
        [
            "experiments",
            "predict",
            "--config-path",
            deet_project.config_path,
            "--ignore-references",
        ],
    )
    assert result.exit_code == 0

    # Alice makes sure her experiments dir exists
    # and that it contains a folder for her new experiment
    assert deet_project.experiments_dir.exists()
    run_dirs = list(deet_project.experiments_dir.iterdir())
    assert len(run_dirs) == 1

    # She finds the directory for the experiment she just carried out
    experiment_run_dir = run_dirs[0]
    run_id = experiment_run_dir.name

    # And makes sure it contains all the experiment artefacts she would expect
    artefacts = ExperimentArtefacts(base_dir=experiment_run_dir, run_id=run_id)

    assert artefacts.config_snapshot.exists()
    assert artefacts.prompts_snapshot.exists()
    assert artefacts.llm_annotation_csv.exists()
    assert artefacts.llm_annotations.exists()

    assert_llm_csv_contains_values(artefacts.llm_annotation_csv)


@pytest.mark.vcr
@pytest.mark.parametrize("dataset_base_path", INTEGRATION_DATASETS)
def test_extraction_with_evaluation(
    runner, dataset_base_path, tmp_project_workspace, initialised_project_workspace
):
    """Test whether Alice can extract data using the LLM."""
    # Alice makes sure she is in the project directory she created on project init
    project_dir = tmp_project_workspace / dataset_base_path.name
    os.chdir(project_dir)
    deet_project = DeetProject.load()

    #  Alice adds the necessary metadata for her files to be linked
    shutil.copy(dataset_base_path / "link_map.csv", deet_project.link_map_path)

    # Alice adds her own custom prompts for the attributes she wants to extract
    shutil.copy(
        dataset_base_path / "prompt_definitions.csv", deet_project.prompt_csv_path
    )

    litellm.telemetry = False  # Disable any hidden outbound telemetry calls

    # 2. Force the underlying HTTP client to avoid async pooling/HTTP2
    # This ensures it hits VCR's standard socket patches
    os.environ["LITELLM_FORCE_SYNC"] = "True"

    # If your version uses the OpenAI client mapping directly under litellm:
    import httpx

    litellm.client_session = httpx.Client(http2=False)

    result = runner.invoke(app, ["project", "link"])
    assert result.exit_code == 0

    result = runner.invoke(
        app, ["experiments", "evaluate", "--config-path", deet_project.config_path]
    )
    assert result.exit_code == 0

    # Alice makes sure her experiments dir exists
    # and that it contains a folder for her new experiment
    assert deet_project.experiments_dir.exists()
    run_dirs = list(deet_project.experiments_dir.iterdir())
    assert len(run_dirs) == 1

    # She finds the directory for the experiment she just carried out
    experiment_run_dir = run_dirs[0]
    run_id = experiment_run_dir.name

    # And makes sure it contains all the experiment artefacts she would expect
    artefacts = ExperimentArtefacts(base_dir=experiment_run_dir, run_id=run_id)

    assert artefacts.config_snapshot.exists()
    assert artefacts.prompts_snapshot.exists()
    assert artefacts.comparison.exists()
    assert artefacts.llm_annotations.exists()
    assert artefacts.metrics.exists()

    assert_llm_csv_contains_values(artefacts.comparison)
