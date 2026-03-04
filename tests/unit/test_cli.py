# tests/test_cli.py

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from deet.scripts.cli import (
    DataExtractionConfig,
    SupportedImportFormat,
    app,
    export_config_template,
    import_gold_standard_data,
    init_prompt_csv,
)

runner = CliRunner()


def test_cli_help() -> None:
    """Make sure cli is callable."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export-config-template" in result.output
    assert "import-gold-standard-data" in result.output


def test_export_default_config_writes_yaml(tmp_path: Path):
    """Make sure default config can be exported to yaml and read back."""
    output_file = tmp_path / "config.yaml"
    export_config_template(output_path=output_file)

    # File should exist
    assert output_file.exists()

    # Contents should load as YAML and match a DataExtractionConfig dict
    data = yaml.safe_load(output_file.read_text())
    config_dict = DataExtractionConfig().model_dump(mode="json")
    assert data == config_dict


def test_import_data_calls_converter_methods():
    """Make sure import_data calls the converter's process_annotation_file() method."""
    fake_result = MagicMock()
    fake_result.annotated_documents = []

    fake_converter = MagicMock()
    fake_converter.process_annotation_file.return_value = fake_result

    with patch.object(
        SupportedImportFormat.DEET,
        "get_annotation_converter",
        return_value=fake_converter,
    ):
        out = import_gold_standard_data(
            gs_data_path=Path("dummy"), gs_data_format=SupportedImportFormat.DEET
        )

    # Should call process_annotation_file with the given path
    fake_converter.process_annotation_file.assert_called_once_with(Path("dummy"))
    # Return value is the processed annotation data
    assert out == fake_result


def test_write_prompt_csv_calls_export(tmp_path: Path):
    """Make sure export_attributes_csv_file() is called by write_prompt_csv."""
    fake_out = MagicMock()
    fake_out.export_attributes_csv_file = MagicMock()

    # Mock import_data to return our fake_out
    with patch("deet.scripts.cli.import_gold_standard_data", return_value=fake_out):
        # Also mock Path.exists to simulate CSV file not existing
        csv_path = tmp_path / "prompt.csv"
        with patch.object(Path, "exists", return_value=False):
            init_prompt_csv(
                gs_data_path=tmp_path,
                gs_data_format=SupportedImportFormat.EPPI_JSON,
                csv_path=csv_path,
            )

    fake_out.export_attributes_csv_file.assert_called_once()

    called_path = fake_out.export_attributes_csv_file.call_args[1]["filepath"]
    assert called_path == csv_path
