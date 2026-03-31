"""Tests for deet/scripts/cli.py."""

from unittest.mock import MagicMock, patch

import pytest
import yaml  # type:ignore[import-untyped]
from typer.testing import CliRunner

from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.logger import logger
from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.cli import app, init_linkage_mapping_file

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
    assert "export-config-template" in result.output


def test_export_default_config_writes_yaml(tmp_path):
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


def test_init_linkage_mapping_file(link_map_path, processed_data):
    """Test init-linkage-mapping-file command."""
    gs_data_path = link_map_path.parent / "dummy.json"

    with patch(
        "deet.processors.converter_register.SupportedImportFormat.get_annotation_converter"
    ) as mock_get_converter:
        mock_converter = mock_get_converter.return_value
        mock_converter.process_annotation_file.return_value = processed_data

        init_linkage_mapping_file(
            gs_data_path=gs_data_path,
            gs_data_format=SupportedImportFormat.EPPI_JSON,
            link_map_path=link_map_path,
        )

    assert link_map_path.exists()


def test_init_prompt_csv_confirmation(gs_data_path, csv_path, mock_converter):
    """Test init-prompt-csv command with confirmation prompts."""
    with patch(
        "deet.data_models.processed_gold_standard_annotations.ProcessedEppiAnnotationData.export_attributes_csv_file"
    ) as mock_export:
        # Case 1: CSV does not exist → export called
        result = runner.invoke(
            app, ["init-prompt-csv", str(gs_data_path), "--csv-path", str(csv_path)]
        )
        assert result.exit_code == 0
        mock_export.assert_called_once_with(filepath=csv_path)

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


def test_link_documents_fulltexts(
    gs_data_path, pdf_dir, tmp_path, mock_converter, link_map_path
):
    """Test link-documents-fulltexts command."""
    output_path = tmp_path / "linked_output"
    output_path.mkdir()

    mock_linked_doc = MagicMock()
    mock_linked_doc.safe_identity.document_id = 12345678

    with patch("deet.processors.linker.DocumentReferenceLinker") as mock_linker_class:
        mock_linker = mock_linker_class.return_value
        mock_linker.link_many_references_parsed_documents.return_value = [
            mock_linked_doc
        ]

        result = runner.invoke(
            app,
            [
                "link-documents-fulltexts",
                str(gs_data_path),
                "--link-map-path",
                str(link_map_path),
                "--pdf-dir",
                str(pdf_dir),
                "--output-path",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        mock_linker.link_many_references_parsed_documents.assert_called_once()
        mock_linked_doc.save.assert_called_once()


def test_extract_data_missing_csv_path(gs_data_path, config_path, mock_converter):
    """Test extract-data fails when prompt-population file but no csv-path provided."""
    with patch("deet.scripts.cli.fail_with_message") as mock_fail:
        mock_fail.side_effect = SystemExit(1)
        result = runner.invoke(
            app,
            [
                "extract-data",
                str(gs_data_path),
                "--config-path",
                str(config_path),
                "--prompt-population",
                "file",
            ],
        )
        assert result.exit_code != 0
        mock_fail.assert_called_once()


@pytest.fixture
def mock_extraction_pipeline(config, out_dir, processed_data):
    """Set up mocks for extract-data command."""
    mock_llm_docs = [MagicMock()]
    mock_run_output = MagicMock()
    mock_run_output.annotated_documents = mock_llm_docs

    with (
        patch(
            "deet.extractors.llm_data_extractor.LLMDataExtractor"
        ) as mock_extractor_class,
        patch(
            "deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator"
        ) as mock_evaluator_class,
        patch("deet.extractors.cli_helpers.init_extraction_run") as mock_init_run,
        patch("deet.extractors.cli_helpers.load_or_init_config") as mock_load_config,
        patch("deet.extractors.cli_helpers.prepare_documents") as mock_prepare_docs,
    ):
        mock_load_config.return_value = config
        run_dir = out_dir / "run_123"
        mock_init_run.return_value = ("run_123", run_dir)
        run_dir.mkdir(parents=True)
        mock_prepare_docs.return_value = processed_data.documents

        mock_extractor = mock_extractor_class.return_value
        mock_extractor.config = config
        mock_extractor.extract_from_documents.return_value = mock_run_output

        mock_evaluator = mock_evaluator_class.return_value

        yield {
            "extractor_class": mock_extractor_class,
            "extractor": mock_extractor,
            "evaluator_class": mock_evaluator_class,
            "evaluator": mock_evaluator,
            "init_run": mock_init_run,
            "load_config": mock_load_config,
            "prepare_docs": mock_prepare_docs,
            "llm_docs": mock_llm_docs,
        }


def test_extract_data_success(
    gs_data_path,
    config_path,
    out_dir,
    linked_doc_path,
    pdf_dir,
    mock_converter,
    mock_extraction_pipeline,
):
    """Test extract-data command runs successfully."""
    result = runner.invoke(
        app,
        [
            "extract-data",
            str(gs_data_path),
            "--config-path",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--linked-document-path",
            str(linked_doc_path),
            "--pdf-dir",
            str(pdf_dir),
        ],
    )
    assert result.exit_code == 0
    mock_extraction_pipeline["extractor"].extract_from_documents.assert_called_once()
    mock_extraction_pipeline["evaluator"].evaluate_llm_annotations.assert_called_once()
    mock_extraction_pipeline["evaluator"].write_metrics_to_csv.assert_called_once()
    mock_extraction_pipeline["evaluator"].export_llm_comparison.assert_called_once()
    mock_extraction_pipeline["evaluator"].display_metrics.assert_called_once()


def test_test_llm_config():
    """Test test-llm-config command."""
    with patch(
        "deet.extractors.llm_data_extractor.LLMDataExtractor"
    ) as mock_extractor_class:
        mock_extractor = mock_extractor_class.return_value
        mock_extractor.extract_from_document.return_value = {"result": "test"}

        result = runner.invoke(app, ["test-llm-config"])
        assert result.exit_code == 0
        mock_extractor.extract_from_document.assert_called_once()


def test_global_options_verbose():
    """Test that verbose flag enables debug-level logging output."""
    result = runner.invoke(app, ["--verbose", "--help"])
    assert result.exit_code == 0
    assert "deet" in result.output


def test_global_options_non_verbose():
    """Test that non-verbose mode runs without debug output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "deet" in result.output


def test_verbose_flag_shows_log_output(gs_data_path, link_map_path, processed_data):
    """Test that verbose flag produces additional log output compared to default."""
    import copy

    # Create fresh copies for each run to avoid mutation issues
    def make_fresh_mock():
        fresh_data = copy.deepcopy(processed_data)
        return MagicMock(process_annotation_file=lambda _: fresh_data)

    # First run without --verbose
    logger.remove()
    with patch.object(
        SupportedImportFormat.EPPI_JSON,
        "get_annotation_converter",
        return_value=make_fresh_mock(),
    ):
        result_non_verbose = runner.invoke(
            app,
            [
                "init-linkage-mapping-file",
                str(gs_data_path),
                "--link-map-path",
                str(link_map_path),
            ],
            catch_exceptions=False,
        )
    assert result_non_verbose.exit_code == 0
    non_verbose_output = result_non_verbose.output

    # delete link map csv we just created
    if link_map_path.exists():
        link_map_path.unlink()

    # Now run with --verbose using fresh data
    logger.remove()
    with patch.object(
        SupportedImportFormat.EPPI_JSON,
        "get_annotation_converter",
        return_value=make_fresh_mock(),
    ):
        result_verbose = runner.invoke(
            app,
            [
                "--verbose",
                "init-linkage-mapping-file",
                str(gs_data_path),
                "--link-map-path",
                str(link_map_path),
            ],
            catch_exceptions=False,
        )
    assert result_verbose.exit_code == 0
    verbose_output = result_verbose.output

    # Verbose mode should produce additional output (e.g., debug logs)
    assert verbose_output != non_verbose_output
    assert len(verbose_output) > len(non_verbose_output)


def test_init_linkage_mapping_file_via_cli(gs_data_path, link_map_path, mock_converter):
    """Test init-linkage-mapping-file via CLI runner."""
    result = runner.invoke(
        app,
        [
            "init-linkage-mapping-file",
            str(gs_data_path),
            "--link-map-path",
            str(link_map_path),
        ],
    )
    assert result.exit_code == 0


def test_extract_data_with_custom_metrics(
    gs_data_path,
    config_path,
    out_dir,
    linked_doc_path,
    pdf_dir,
    mock_converter,
    mock_extraction_pipeline,
):
    """Test extract-data with custom evaluation metrics."""
    result = runner.invoke(
        app,
        [
            "extract-data",
            str(gs_data_path),
            "--config-path",
            str(config_path),
            "--out-dir",
            str(out_dir),
            "--linked-document-path",
            str(linked_doc_path),
            "--pdf-dir",
            str(pdf_dir),
            "--custom-evaluation-metrics",
            "brier_score_loss",
            "--custom-evaluation-metrics",
            "cohen_kappa_score",
        ],
    )
    assert result.exit_code == 0
    call_kwargs = mock_extraction_pipeline["evaluator_class"].call_args[1]
    assert call_kwargs["custom_metrics"] == ["brier_score_loss", "cohen_kappa_score"]


def test_extract_data_with_prompt_population(
    gs_data_path,
    config_path,
    csv_path,
    out_dir,
    linked_doc_path,
    pdf_dir,
    processed_data,
    config,
):
    """Test extract-data with prompt population from file."""
    csv_path.write_text("attribute_id,prompt\n1,test prompt")

    mock_processed = MagicMock()
    mock_processed.documents = processed_data.documents
    mock_processed.attributes = processed_data.attributes
    mock_processed.annotated_documents = processed_data.annotated_documents

    mock_llm_docs = [MagicMock()]
    mock_run_output = MagicMock()
    mock_run_output.annotated_documents = mock_llm_docs

    with (
        patch.object(
            SupportedImportFormat.EPPI_JSON,
            "get_annotation_converter",
            return_value=MagicMock(process_annotation_file=lambda _: mock_processed),
        ),
        patch(
            "deet.extractors.llm_data_extractor.LLMDataExtractor"
        ) as mock_extractor_class,
        patch("deet.evaluators.gold_standard_llm_evaluator.GoldStandardLLMEvaluator"),
        patch("deet.extractors.cli_helpers.init_extraction_run") as mock_init_run,
        patch("deet.extractors.cli_helpers.load_or_init_config") as mock_load_config,
        patch("deet.extractors.cli_helpers.prepare_documents") as mock_prepare_docs,
    ):
        mock_load_config.return_value = config
        run_dir = out_dir / "run_123"
        mock_init_run.return_value = ("run_123", run_dir)
        run_dir.mkdir(parents=True)
        mock_prepare_docs.return_value = processed_data.documents

        mock_extractor = mock_extractor_class.return_value
        mock_extractor.config = config
        mock_extractor.extract_from_documents.return_value = mock_run_output

        result = runner.invoke(
            app,
            [
                "extract-data",
                str(gs_data_path),
                "--config-path",
                str(config_path),
                "--out-dir",
                str(out_dir),
                "--linked-document-path",
                str(linked_doc_path),
                "--pdf-dir",
                str(pdf_dir),
                "--prompt-population",
                "file",
                "--csv-path",
                str(csv_path),
            ],
        )
        assert result.exit_code == 0
        mock_processed.populate_custom_prompts.assert_called_once()
