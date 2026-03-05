# tests/test_cli.py

from pathlib import Path

import yaml
from typer.testing import CliRunner

from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.scripts.cli import (
    app,
    export_config_template,
)

runner = CliRunner()


def test_cli_help() -> None:
    """Make sure cli is callable."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export-config-template" in result.output


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
