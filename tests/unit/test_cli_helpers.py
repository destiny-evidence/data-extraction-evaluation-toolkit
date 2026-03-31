"""Tests for deet/extractors/cli_helpers.py."""

from unittest.mock import MagicMock, patch

import pytest
import yaml  # type:ignore[import-untyped]

from deet.data_models.documents import ContextType, Document
from deet.extractors.cli_helpers import (
    init_extraction_run,
    load_or_init_config,
    prepare_documents,
)
from deet.extractors.llm_data_extractor import DataExtractionConfig


@pytest.fixture
def config():
    """Create a default DataExtractionConfig."""
    return DataExtractionConfig()


@pytest.fixture
def config_path(tmp_path, config):
    """Create a config YAML file."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


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
def mock_documents():
    """Create mock documents."""
    doc1 = MagicMock(spec=Document)
    doc2 = MagicMock(spec=Document)
    return [doc1, doc2]


def test_load_or_init_config_file_exists(config_path, config):
    """Test loading config from existing file."""
    loaded_config = load_or_init_config(config_path)

    assert isinstance(loaded_config, DataExtractionConfig)
    assert loaded_config.model_dump() == config.model_dump()


def test_load_or_init_config_file_doesnt_exist(tmp_path):
    """Test initializing default config when file doesn't exist."""
    non_existent_path = tmp_path / "non_existent_config.yaml"

    with patch("deet.extractors.cli_helpers.echo_and_log") as mock_echo:
        loaded_config = load_or_init_config(non_existent_path)

    assert isinstance(loaded_config, DataExtractionConfig)
    assert loaded_config.model_dump() == DataExtractionConfig().model_dump()
    mock_echo.assert_called_once()
    assert "does not exist" in mock_echo.call_args[0][0]


def test_load_or_init_config_file_invalid(tmp_path):
    """When our config yaml file is invalid."""
    invalid_config_path = tmp_path / "invalid_config.yaml"
    invalid_config_path.write_text("invalid: yaml: content: [")

    with pytest.raises(yaml.YAMLError):
        load_or_init_config(invalid_config_path)


def test_init_extraction_run(tmp_path):
    """Ensure it creates the folder; ensure it creates deet.log."""
    out_dir = tmp_path / "experiments"
    out_dir.mkdir()
    run_name = "test_run"

    with patch("deet.extractors.cli_helpers.logger") as mock_logger:
        extraction_run_id, experiment_out_dir = init_extraction_run(out_dir, run_name)

    # run ID format contains timestamp and run name
    assert run_name in extraction_run_id
    assert "_" in extraction_run_id  # timestamp separator

    # check experiment directory was created
    assert experiment_out_dir.exists()
    assert experiment_out_dir.is_dir()
    assert experiment_out_dir.parent == out_dir

    # check logger.add was called with log file path
    mock_logger.add.assert_called_once()
    log_path = mock_logger.add.call_args[0][0]
    assert log_path == experiment_out_dir / "deet.log"


def test_prepare_documents_context_type_abstract(mock_documents, config, tmp_path):
    """Return just the documents when context type is abstract only."""
    config.default_context_type = ContextType.ABSTRACT_ONLY
    linked_doc_path = tmp_path / "linked_documents"
    pdf_dir = tmp_path / "pdfs"

    result = prepare_documents(
        documents=mock_documents,
        config=config,
        linked_document_path=linked_doc_path,
        pdf_dir=pdf_dir,
        link_map_path=None,
    )

    assert result == mock_documents


def test_prepare_documents_context_full_doc_linked_exists(config, tmp_path):
    """Load linked documents when they already exist."""
    config.default_context_type = ContextType.FULL_DOCUMENT
    linked_doc_path = tmp_path / "linked_documents"
    linked_doc_path.mkdir()
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    # Create some mock linked document files
    (linked_doc_path / "doc1.json").write_text("{}")
    (linked_doc_path / "doc2.json").write_text("{}")

    mock_loaded_doc = MagicMock(spec=Document)

    with patch.object(Document, "load", return_value=mock_loaded_doc) as mock_load:
        result = prepare_documents(
            documents=[],
            config=config,
            linked_document_path=linked_doc_path,
            pdf_dir=pdf_dir,
            link_map_path=None,
        )

    assert len(result) == 2
    assert mock_load.call_count == 2


def test_prepare_documents_unsupported_context_type(config, tmp_path, mock_documents):
    """Test that unsupported context type fails with message."""
    # Create a mock unsupported context type
    config.default_context_type = MagicMock()
    config.default_context_type.__eq__ = lambda _, __: False

    linked_doc_path = tmp_path / "linked_documents"
    pdf_dir = tmp_path / "pdfs"

    with patch("deet.extractors.cli_helpers.fail_with_message") as mock_fail:
        mock_fail.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            prepare_documents(
                documents=mock_documents,
                config=config,
                linked_document_path=linked_doc_path,
                pdf_dir=pdf_dir,
                link_map_path=None,
            )

        mock_fail.assert_called_once()
        assert "not supported" in mock_fail.call_args[0][0]


def test_prepare_documents_failed_to_link(config, tmp_path, mock_documents):
    """Test failure when no linked documents could be found or created."""
    config.default_context_type = ContextType.FULL_DOCUMENT
    linked_doc_path = tmp_path / "linked_documents"
    # Don't create the directory
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    with (
        patch("deet.extractors.cli_helpers.echo_and_log"),
        patch(
            "deet.extractors.cli_helpers.DocumentReferenceLinker"
        ) as mock_linker_class,
        patch("deet.extractors.cli_helpers.fail_with_message") as mock_fail,
    ):
        mock_linker = mock_linker_class.return_value
        # Return empty list - no documents could be linked
        mock_linker.link_many_references_parsed_documents.return_value = []
        mock_fail.side_effect = SystemExit(1)

        with pytest.raises(SystemExit):
            prepare_documents(
                documents=mock_documents,
                config=config,
                linked_document_path=linked_doc_path,
                pdf_dir=pdf_dir,
                link_map_path=None,
            )

        mock_fail.assert_called_once()
        assert any(
            msg in mock_fail.call_args[0][0]
            for msg in (
                "no linked documents could be found",
                "Linked document path does not exist",
            )
        )
