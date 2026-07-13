"""Tests for deet/scripts/cli.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml  # type:ignore[import-untyped]
from typer.testing import CliRunner

from deet.data_models.project import DeetProject
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.cli import app
from deet.scripts.typer_context import CLIState, project_required
from deet.settings import DataExtractionSettings
from deet.utils.text import slugify

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
def command_with_project_required(typer_context: typer.Context):
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
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
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
        runner.isolated_filesystem(),
        patch("deet.data_models.project.DeetProject.load", return_value=fake_project),
        patch("deet.scripts.project_utils.inquirer.confirm") as mock_confirm,
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        Path("project.yaml").touch()  # an existing project in the current dir
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
        runner.isolated_filesystem(),
        patch("deet.data_models.project.DeetProject.load", return_value=fake_project),
        patch("deet.scripts.project_utils.inquirer.confirm") as mock_confirm,
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        Path("project.yaml").touch()  # an existing project in the current dir
        mock_confirm.return_value.execute.return_value = True
        mock_wizard.side_effect = [new_project, fake_settings]

        result = runner.invoke(app, ["project", "init"])

        assert result.exit_code == 0
        assert mock_wizard.call_count == 2
        new_project.setup.assert_called_once()
        fake_settings.dump_to_env.assert_called_once()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("My Project", "my-project"),
        ("  Climate & Health v2 ", "climate-health-v2"),
        ("already-slug", "already-slug"),
    ],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


def test_slugify_rejects_unusable_name():
    with pytest.raises(ValueError, match="Could not derive"):
        slugify("---")


def test_new_project_creates_directory_and_anchors():
    fake_project = MagicMock(spec=DeetProject)
    fake_settings = MagicMock(spec=DataExtractionSettings)

    with (
        runner.isolated_filesystem() as td,
        patch(
            "deet.data_models.project.DeetProject.load",
            side_effect=FileNotFoundError,
        ),
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        mock_wizard.side_effect = [fake_project, fake_settings]

        result = runner.invoke(app, ["project", "new", "--name", "My Project"])
        target = Path(td) / "my-project"

        assert result.exit_code == 0
        assert target.exists()
        fake_project.anchor_to.assert_called_once_with(target)
        fake_project.setup.assert_called_once()
        fake_settings.dump_to_env.assert_called_once()


def test_new_project_prompts_for_name_when_omitted():
    fake_project = MagicMock(spec=DeetProject)
    fake_settings = MagicMock(spec=DataExtractionSettings)

    with (
        runner.isolated_filesystem() as td,
        patch(
            "deet.data_models.project.DeetProject.load",
            side_effect=FileNotFoundError,
        ),
        patch(
            "deet.scripts.project_utils.inquire_pydantic_field",
            return_value="My Project",
        ) as mock_name_prompt,
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        mock_wizard.side_effect = [fake_project, fake_settings]

        result = runner.invoke(app, ["project", "new"])
        target = Path(td) / "my-project"

        assert result.exit_code == 0
        mock_name_prompt.assert_called_once()  # name collected interactively
        assert target.exists()
        fake_project.anchor_to.assert_called_once_with(target)


def test_new_project_headless_with_args():
    with (
        runner.isolated_filesystem() as td,
        patch(
            "deet.data_models.project.DeetProject.load",
            side_effect=FileNotFoundError,
        ),
        patch("deet.data_models.project.DeetProject.setup", return_value=None),
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
    ):
        Path("references.json").touch()
        result = runner.invoke(
            app,
            ["project", "new", "--name", "My Project", "-d", "references.json"],
        )
        target = Path(td) / "my-project"

        assert result.exit_code == 0
        assert target.exists()
        mock_wizard.assert_not_called()


def _wizard_result_like(project):
    """Build a MagicMock standing in for a wizard-produced project."""
    updated = MagicMock(spec=DeetProject)
    updated.name = project.name
    updated.gold_standard_data_format = project.gold_standard_data_format
    updated.gold_standard_data_path = project.gold_standard_data_path
    return updated


def test_edit_project_full_writes_yaml_without_setup(
    valid_project_data, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    project = DeetProject(**valid_project_data)
    updated = _wizard_result_like(project)
    settings_out = MagicMock(spec=DataExtractionSettings)
    settings_out.azure_api_key = None
    settings_out.azure_api_base = None

    with (
        patch("deet.data_models.project.DeetProject.load", return_value=project),
        patch(
            "deet.scripts.project_utils.run_model_wizard",
            side_effect=[updated, settings_out],
        ) as mock_wizard,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        result = runner.invoke(app, ["project", "edit"])

    assert result.exit_code == 0
    assert mock_wizard.call_count == 2  # project fields + credentials
    updated.anchor_to.assert_called_once_with(project.root)
    assert updated.created_at == project.created_at  # preserved, not reset
    updated.dump_to_yaml.assert_called_once()
    updated.setup.assert_not_called()  # artefacts NOT regenerated
    # credentials are written to the project's own .env
    settings_out.dump_to_env.assert_called_once_with(target_path=tmp_path / ".env")


def test_edit_single_field_only_prompts_that_field(
    valid_project_data, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    project = DeetProject(**valid_project_data)
    updated = _wizard_result_like(project)

    with (
        patch("deet.data_models.project.DeetProject.load", return_value=project),
        patch(
            "deet.scripts.project_utils.inquire_pydantic_field",
            return_value="new_pdfs",
        ),
        patch(
            "deet.data_models.project.DeetProject.model_validate", return_value=updated
        ) as mock_validate,
        patch("deet.scripts.project_utils.run_model_wizard") as mock_wizard,
        patch("deet.scripts.project_utils.console.clear"),
    ):
        result = runner.invoke(app, ["project", "edit", "pdf_dir"])

    assert result.exit_code == 0
    mock_wizard.assert_not_called()
    updated.dump_to_yaml.assert_called_once()
    updated.setup.assert_not_called()
    merged = mock_validate.call_args.args[0]
    assert merged["pdf_dir"] == "new_pdfs"


def test_edit_rejects_unknown_field(valid_project_data, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    project = DeetProject(**valid_project_data)

    with patch("deet.data_models.project.DeetProject.load", return_value=project):
        result = runner.invoke(app, ["project", "edit", "bogus"])

    assert result.exit_code == 1
    assert "not an editable field" in result.stderr


def test_edit_warns_when_data_source_changes(valid_project_data, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    project = DeetProject(**valid_project_data)
    updated = _wizard_result_like(project)
    updated.gold_standard_data_path = Path("different.json")  # changed
    settings_out = MagicMock(spec=DataExtractionSettings)
    settings_out.azure_api_key = None
    settings_out.azure_api_base = None

    with (
        patch("deet.data_models.project.DeetProject.load", return_value=project),
        patch(
            "deet.scripts.project_utils.run_model_wizard",
            side_effect=[updated, settings_out],
        ),
        patch("deet.scripts.project_utils.notify") as mock_notify,
        patch("deet.scripts.project_utils.continue_after_key"),
        patch("deet.scripts.project_utils.console.clear"),
    ):
        result = runner.invoke(app, ["project", "edit"])

    assert result.exit_code == 0
    assert any(
        "regenerate" in str(call.args[0]).lower() for call in mock_notify.call_args_list
    )


def test_init_project_noninteractive(tmp_path):
    data_file = tmp_path / "references.json"
    data_file.touch()

    with (
        patch("deet.data_models.project.DeetProject.load", return_value=None),
        patch("deet.data_models.project.DeetProject.setup", return_value=None),
    ):
        result = runner.invoke(app, ["project", "init", "-d", str(data_file)])

    assert result.exit_code == 0


def test_init_project_noninteractive_fails_with_insufficient_args(tmp_path):
    with (
        patch("deet.data_models.project.DeetProject.load", return_value=None),
        patch("deet.data_models.project.DeetProject.setup", return_value=None),
    ):
        result = runner.invoke(app, ["project", "init", "-p", str(tmp_path)])

    assert "is required to create a project" in result.output
    assert result.exit_code == 1


def test_init_project_noninteractive_no_overwrite():
    with (
        runner.isolated_filesystem(),
        patch("deet.data_models.project.DeetProject.load", return_value=None),
        patch("deet.data_models.project.DeetProject.setup", return_value=None),
    ):
        Path("project.yaml").touch()  # an existing project in the current dir
        Path("references.json").touch()
        result = runner.invoke(app, ["project", "init", "-d", "references.json"])

        assert "already exists" in result.stderr
        assert result.exit_code == 1


def test_init_project_noninteractive_force_overwrite():
    with (
        runner.isolated_filesystem(),
        patch("deet.data_models.project.DeetProject.load", return_value=None),
        patch("deet.data_models.project.DeetProject.setup", return_value=None),
    ):
        Path("project.yaml").touch()  # existing project, overridden by --force
        Path("references.json").touch()
        result = runner.invoke(app, ["project", "init", "-d", "references.json", "-f"])

        assert result.exit_code == 0


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
        patch("deet.extractors.cli_helpers.LLMDataExtractor") as mock_extractor_cls,
        patch("deet.extractors.cli_helpers.continue_after_key"),
        patch("deet.extractors.cli_helpers.console.clear"),
        patch("deet.extractors.cli_helpers.prepare_documents") as mock_prepare,
        patch(
            "deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator"
        ) as mock_evaluator_cls,
    ):
        mock_prepare.return_value = ([], {})
        mock_loader.return_value = mock_project
        fake_config = DataExtractionConfig()
        mock_wizard.return_value = fake_config

        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.config = fake_config
        mock_run_output = MagicMock()
        mock_run_output.annotated_documents = mock_processed_data.annotated_documents
        mock_run_output.metadata.model_dump_json.return_value = "{}"
        mock_extractor.extract_from_documents.return_value = mock_run_output

        mock_evaluator = mock_evaluator_cls.return_value

        result = runner.invoke(app, ["experiments", "evaluate"], obj=state)

    assert result.exit_code == 0
    mock_extractor.extract_from_documents.assert_called_once()
    mock_evaluator.evaluate_llm_annotations.assert_called_once()
    mock_evaluator.write_metrics_to_csv.assert_called_once()
    mock_evaluator.export_llm_comparison.assert_called_once()
    mock_evaluator.display_metrics.assert_called_once()


def test_test_llm_config():
    mock_cfg = MagicMock(spec=DataExtractionConfig)

    with (
        patch(
            "deet.extractors.cli_helpers.load_config_from_typer_context"
        ) as mock_load,
        patch(
            "deet.extractors.llm_data_extractor.LLMDataExtractor"
        ) as mock_extractor_cls,
    ):
        mock_load.return_value = mock_cfg

        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.extract_from_document.return_value = MagicMock(annotations=[1])

        result = runner.invoke(app, ["project", "test-llm-config"])

    assert result.exit_code == 0


@pytest.mark.parametrize(
    "command",
    [
        "extract-data",
        "export-config-template",
        "init-linkage-mapping-file",
        "link-documents-fulltexts",
        "init-prompt-csv",
        "test-llm-config",
    ],
)
def test_deprecated_commands_return_deprecation_warning(command):
    result = runner.invoke(app, [command])
    assert "deprecated" in result.stdout.lower()
    assert command in result.stdout.lower()
