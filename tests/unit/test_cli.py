# tests/test_cli.py

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.cli import (
    app,
    init_linkage_mapping_file,
)

runner = CliRunner()

pytest_plugins = ["tests.unit.test_eppi"]


def test_cli_help() -> None:
    """Make sure cli is callable."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export-config-template" in result.output


def test_export_default_config_writes_yaml(tmp_path: Path):
    """Make sure default config can be exported to yaml and read back."""
    output_file = tmp_path / "config.yaml"

    result = runner.invoke(
        app, ["export-config-template", "--output-path", str(output_file)]
    )
    assert result.exit_code == 0
    assert output_file.exists()

    # Contents should load as YAML and match a DataExtractionConfig dict
    data = yaml.safe_load(output_file.read_text())
    config_dict = DataExtractionConfig().model_dump(mode="json")
    assert data == config_dict


def test_init_linkage_mapping_file(tmp_path: Path, processed_data):
    """Test init-linkage-mapping-file command."""
    gs_data_path = tmp_path / "dummy.json"
    link_map_path = tmp_path / "link_map.csv"

    with patch(
        "deet.processors.converter_register.SupportedImportFormat.get_annotation_converter"
    ) as mock_get_converter:
        mock_converter = mock_get_converter.return_value
        # Patch process_annotation_file to return our fixture
        mock_converter.process_annotation_file.return_value = processed_data

        # Call the CLI function directly
        init_linkage_mapping_file(
            gs_data_path=gs_data_path,
            gs_data_format=SupportedImportFormat.EPPI_JSON,
            link_map_path=link_map_path,
        )

    # Assert the file now exists
    assert link_map_path.exists()


def test_init_prompt_csv_confirmation(tmp_path, processed_data):
    gs_data_path = tmp_path / "dummy.json"
    gs_data_path.write_text("{}")  # dummy input

    csv_path = tmp_path / "prompt.csv"

    with (
        patch.object(
            SupportedImportFormat.EPPI_JSON,
            "get_annotation_converter",
            return_value=MagicMock(process_annotation_file=lambda _: processed_data),
        ),
        patch(
            "deet.data_models.processed_gold_standard_annotations.ProcessedEppiAnnotationData.export_attributes_csv_file"
        ) as mock_export,
    ):
        # Case 1: CSV does not exist → export called
        result = runner.invoke(
            app, ["init-prompt-csv", str(gs_data_path), "--csv-path", str(csv_path)]
        )
        assert result.exit_code == 0
        mock_export.assert_called_once_with(filepath=csv_path)

        # Reset mock for next case
        mock_export.reset_mock()

        # Case 2: CSV exists → user confirms overwrite → export called
        csv_path.write_text("existing content")
        result = runner.invoke(
            app,
            ["init-prompt-csv", str(gs_data_path), "--csv-path", str(csv_path)],
            input="y\n",
        )
        assert result.exit_code == 0
        mock_export.assert_called_once_with(filepath=csv_path)

        # Reset mock for next case
        mock_export.reset_mock()

        # Case 3: CSV exists → user declines overwrite → typer.Abort
        csv_path.write_text("existing content")
        result = runner.invoke(
            app,
            ["init-prompt-csv", str(gs_data_path), "--csv-path", str(csv_path)],
            input="n\n",
        )
        assert result.exit_code != 0
        mock_export.assert_not_called()
