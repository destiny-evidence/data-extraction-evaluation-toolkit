"""Tests for deet/scripts/cli.py."""

from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml  # type:ignore[import-untyped]
from typer.testing import CliRunner

from deet.data_models.project import DeetProject
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.cli import app
from deet.scripts.context import CLIState, project_required
from deet.settings import DataExtractionSettings

runner = CliRunner()

pytest_plugins = ["tests.unit.test_eppi"]


@pytest.fixture
def gs_data_path(tmp_path):
    """Create a dummy gold standard data file."""
    path = tmp_path / "dummy.json"
    path.write_text("{}")
    return path


@pytest.fixture
def config(tmp_path):
    """Create a default DataExtractionConfig."""
    return DataExtractionConfig()


@pytest.fixture
def config_path(tmp_path, config):
    """Create a config YAML file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


@pytest.fixture
def csv_path(tmp_path):
    """Create a CSV path for prompts."""
    return tmp_path / "prompts.csv"


@pytest.fixture
def out_dir(tmp_path):
    """Create an output directory for experiments."""
    return tmp_path / "experiments"


@pytest.fixture
def linked_doc_path(tmp_path):
    """Create a linked documents directory."""
    path = tmp_path / "linked_documents"
    path.mkdir()
    return path


@pytest.fixture
def pdf_dir(tmp_path):
    """Create a PDF directory."""
    path = tmp_path / "pdfs"
    path.mkdir()
    return path


@pytest.fixture
def link_map_path(tmp_path):
    """Create a link map path."""
    return tmp_path / "link_map.csv"


@pytest.fixture
def mock_converter(processed_data):
    """Create a mock annotation converter."""
    with patch.object(
        SupportedImportFormat.EPPI_JSON,
        "get_annotation_converter",
        return_value=MagicMock(process_annotation_file=lambda _: processed_data),
    ) as mock:
        yield mock


def test_cli_help():
    """Make sure cli is callable."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data extraction" in result.output


app_mock = typer.Typer()


@app_mock.command()
@project_required
def command_with_project_required(ctx: typer.Context):
    typer.echo("This command works")


def test_project_required_blocks_when_no_project():
    state = CLIState()
    state.project = None

    result = runner.invoke(app_mock, obj=state)

    assert result.exit_code != 0
    assert (
        "This command must be run from a directory that contains a project"
        in result.stderr
    )
    assert "This command works" not in result.stdout


def test_project_required_allows_when_project_exists():
    state = CLIState()
    state.project = MagicMock(spec=DeetProject)

    result = runner.invoke(app_mock, obj=state)

    assert result.exit_code == 0
    assert (
        "This command must be run from a directory that contains a project"
        not in result.stderr
    )
    assert "This command works" in result.stdout


def test_init_project_initialises_in_emptydir():
    fake_project = MagicMock(spec=DeetProject)
    fake_settings = MagicMock(spec=DataExtractionSettings)

    with (
        patch("deet.data_models.project.DeetProject.load") as mock_load,
        patch("deet.scripts.commands.project.run_model_wizard") as mock_wizard,
        patch("deet.scripts.commands.project.continue_after_key"),
        patch("deet.scripts.commands.project.console.clear"),
    ):
        mock_load.side_effect = FileNotFoundError
        mock_wizard.side_effect = [fake_project, fake_settings]

        result = runner.invoke(app, ["project", "init"])

    assert result.exit_code == 0
    assert mock_wizard.call_count == 2
    fake_project.setup.assert_called_once()
    fake_settings.dump_to_env.assert_called_once()


def test_init_project_aborts_no_overwrite():
    fake_project = MagicMock(spec=DeetProject)
    fake_project.name = "Existing project"
    fake_settings = MagicMock(spec=DataExtractionSettings)

    with (
        patch("deet.data_models.project.DeetProject.load") as mock_load,
        patch("deet.scripts.commands.project.inquirer.confirm") as mock_confirm,
        patch("deet.scripts.commands.project.run_model_wizard") as mock_wizard,
        patch("deet.scripts.commands.project.continue_after_key"),
        patch("deet.scripts.commands.project.console.clear"),
    ):
        mock_load.return_value = fake_project
        mock_confirm.return_value.execute.return_value = False
        mock_wizard.side_effect = [fake_project, fake_settings]

        result = runner.invoke(app, ["project", "init"])

    assert result.exit_code == 1
    assert mock_wizard.call_count == 0
    fake_project.setup.assert_not_called()
    fake_settings.dump_to_env.assert_not_called()


def test_init_project_overwrites_after_confirm():
    fake_project = MagicMock(spec=DeetProject)
    fake_project.name = "Existing project"
    fake_settings = MagicMock(spec=DataExtractionSettings)
    new_project = MagicMock(spec=DeetProject)

    with (
        patch("deet.data_models.project.DeetProject.load") as mock_load,
        patch("deet.scripts.commands.project.inquirer.confirm") as mock_confirm,
        patch("deet.scripts.commands.project.run_model_wizard") as mock_wizard,
        patch("deet.scripts.commands.project.continue_after_key"),
        patch("deet.scripts.commands.project.console.clear"),
    ):
        mock_load.return_value = fake_project
        mock_confirm.return_value.execute.return_value = True
        mock_wizard.side_effect = [new_project, fake_settings]

        result = runner.invoke(app, ["project", "init"])

    assert result.exit_code == 0
    assert mock_wizard.call_count == 2
    new_project.setup.assert_called_once()
    fake_settings.dump_to_env.assert_called_once()


def test_link(valid_project_data):
    sample_project = DeetProject.model_validate(valid_project_data)

    mock_linked_doc = MagicMock()
    mock_linked_doc.safe_identity.document_id = 12345678

    with (
        patch("deet.data_models.project.DeetProject.load") as mock_load,
        patch("deet.processors.linker.DocumentReferenceLinker") as mock_linker_class,
    ):
        mock_load.return_value = sample_project
        mock_linker = mock_linker_class.return_value
        mock_linker.link_many_references_parsed_documents.return_value = [
            mock_linked_doc
        ]

        result = runner.invoke(app, ["project", "link"])

        assert result.exit_code == 0
        mock_linker.link_many_references_parsed_documents.assert_called_once()
        mock_linked_doc.save.assert_called_once()


def test_extract_happy_path(tmp_path):
    exp_dir = tmp_path / "experiments"
    mock_project = MagicMock(spec=DeetProject)
    mock_project.experiments_dir = exp_dir
    mock_project.pdf_dir = tmp_path / "pdfs"

    mock_processed_data = MagicMock()
    mock_processed_data.attributes = [1]
    mock_processed_data.documents = []
    mock_processed_data.annotated_documents = []

    mock_project.process_data.return_value = mock_processed_data

    state = CLIState()
    state.project = mock_project

    with (
        patch("deet.data_models.project.DeetProject.load") as mock_loader,
        patch("deet.extractors.cli_helpers.run_model_wizard") as mock_wizard,
        patch(
            "deet.extractors.llm_data_extractor.LLMDataExtractor"
        ) as mock_extractor_cls,
        patch("deet.extractors.cli_helpers.continue_after_key"),
        patch("deet.extractors.cli_helpers.console.clear"),
        patch("deet.extractors.cli_helpers.prepare_documents") as mock_prepare,
        patch(
            "deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator"
        ) as mock_evaluator_cls,
    ):
        mock_prepare.return_value = []
        mock_loader.return_value = mock_project
        fake_config = DataExtractionConfig()
        mock_wizard.return_value = fake_config

        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.config = fake_config
        mock_run_output = MagicMock()
        mock_run_output.annotated_documents = mock_processed_data.annotated_documents
        mock_extractor.extract_from_documents.return_value = mock_run_output

        mock_evaluator = mock_evaluator_cls.return_value

        result = runner.invoke(app, ["run", "extract"], obj=state)

    assert result.exit_code == 0
    mock_extractor.extract_from_documents.assert_called_once()
    mock_evaluator.evaluate_llm_annotations.assert_called_once()
    mock_evaluator.write_metrics_to_csv.assert_called_once()
    mock_evaluator.export_llm_comparison.assert_called_once()
    mock_evaluator.display_metrics.assert_called_once()


def test_test_llm_config():
    mock_cfg = MagicMock(spec=DataExtractionConfig)

    with (
        patch("deet.extractors.cli_helpers.load_config_from_context") as mock_load,
        patch(
            "deet.extractors.llm_data_extractor.LLMDataExtractor"
        ) as mock_extractor_cls,
    ):
        mock_load.return_value = mock_cfg

        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_from_document.return_value = MagicMock(annotations=[1])

        result = runner.invoke(app, ["project", "test-llm-config"])

    assert result.exit_code == 0
