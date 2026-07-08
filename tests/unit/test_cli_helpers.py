"""Tests for deet/extractors/cli_helpers.py."""

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml  # type:ignore[import-untyped]
from destiny_sdk.references import ReferenceFileInput

from deet.data_models.documents import ContextType, Document
from deet.data_models.extraction import ExtractionRunMetadata, ExtractionRunOutput
from deet.extractors.cli_helpers import (
    load_config_from_typer_context,
    prepare_documents,
    run_extraction_pipeline,
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
    mock_typer_context = MagicMock()
    loaded_config = load_config_from_typer_context(mock_typer_context, config_path)

    assert isinstance(loaded_config, DataExtractionConfig)
    assert loaded_config.model_dump() == config.model_dump()


def test_load_or_init_config_file_exists_invalid_yaml(tmp_path):
    """Test loading config from existing file."""
    mock_typer_context = MagicMock()
    config_path = tmp_path / "bad_yaml.yaml"
    config_path.write_text("model_name: gpt-4\n  invalid_indent: true")
    with patch("deet.extractors.cli_helpers.fail_with_message") as mock_fail:
        load_config_from_typer_context(mock_typer_context, config_path)

    assert "YAML Syntax Error" in mock_fail.call_args[0][0]


def test_load_or_init_config_file_exists_invalid_config(tmp_path):
    """Test loading config from existing file."""
    mock_typer_context = MagicMock()
    config_path = tmp_path / "bad_yaml.yaml"
    config_path.write_text("provider: unsupported_provider")
    with patch("deet.extractors.cli_helpers.fail_with_message") as mock_fail:
        load_config_from_typer_context(mock_typer_context, config_path)

    assert "Config validation error" in mock_fail.call_args[0][0]


def test_load_or_init_config_file_doesnt_exist(tmp_path):
    """Test initializing default config when file doesn't exist."""
    non_existent_path = tmp_path / "non_existent_config.yaml"
    mock_typer_context = MagicMock()

    with patch("deet.extractors.cli_helpers.fail_with_message") as mock_fail:
        load_config_from_typer_context(mock_typer_context, non_existent_path)

    assert "file not found" in mock_fail.call_args[0][0]


def test_load_or_init_config_file_doesnt_exist_reverts_project(config_path, config):
    """Test initializing default config when file doesn't exist."""
    mock_project = MagicMock()
    mock_project.config_path = config_path
    mock_typer_context = MagicMock()
    mock_typer_context.obj.project = mock_project

    with (
        patch("deet.extractors.cli_helpers.run_model_wizard") as mock_wizard,
        patch("deet.extractors.cli_helpers.continue_after_key"),
        patch("deet.extractors.cli_helpers.console.clear"),
    ):
        mock_wizard.return_value = config
        loaded_config = load_config_from_typer_context(mock_typer_context, None)

    assert isinstance(loaded_config, DataExtractionConfig)
    assert loaded_config.model_dump() == config.model_dump()


def test_run_extraction_pipeline_writes_run_metadata(tmp_path, config):
    """run_extraction_pipeline should persist run metadata (cost/tokens) to disk."""
    exp_dir = tmp_path / "experiments"

    mock_project = MagicMock()
    mock_project.experiments_dir = exp_dir
    mock_project.pdf_dir = tmp_path / "pdfs"
    mock_project.load_evaluation_strategy.return_value.get_active_ids.return_value = [1]

    mock_processed_data = MagicMock()
    mock_processed_data.attributes = [1]
    mock_processed_data.documents = [MagicMock()]
    mock_project.process_data.return_value = mock_processed_data

    mock_typer_context = MagicMock()
    mock_typer_context.obj.project = mock_project

    run_metadata = ExtractionRunMetadata(
        model="gpt-4o-mini",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=0.0123,
        per_document_tokens={"doc-1": {"input_tokens": 100, "output_tokens": 50}},
    )
    run_output = ExtractionRunOutput(annotated_documents=[], metadata=run_metadata)

    with (
        patch(
            "deet.extractors.cli_helpers.load_config_from_typer_context",
            return_value=config,
        ),
        patch("deet.extractors.cli_helpers.LLMDataExtractor") as mock_extractor_cls,
        patch("deet.extractors.cli_helpers.prepare_documents", return_value=[]),
    ):
        mock_extractor = mock_extractor_cls.return_value
        mock_extractor.config = config
        mock_extractor.extract_from_documents.return_value = run_output

        result_output, _, experiment_artefacts = run_extraction_pipeline(
            typer_context=mock_typer_context,
            prompt_population=None,
        )

    assert result_output is run_output

    metadata_path = experiment_artefacts.run_metadata
    assert metadata_path.name == "run_metadata.json"
    assert metadata_path.exists()

    written = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert written["model"] == "gpt-4o-mini"
    assert written["total_input_tokens"] == 100
    assert written["total_output_tokens"] == 50
    assert written["total_cost_usd"] == 0.0123


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

    documents = [
        Document(document_id=1234, name="1234", citation=ReferenceFileInput()),
        Document(document_id=3456, name="1234", citation=ReferenceFileInput()),
    ]
    # Create some mock linked document files
    for document in documents:
        document.init_document_identity()
        (
            (linked_doc_path / f"{document.safe_identity.document_id}.json").write_text(
                "{}"
            )
        )

    mock_loaded_doc = MagicMock(spec=Document)

    with patch.object(Document, "load", return_value=mock_loaded_doc) as mock_load:
        result = prepare_documents(
            documents=documents,
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
        patch("deet.extractors.cli_helpers.notify"),
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
                "No link map supplied",
                "no linked documents could be found",
                "Linked document path does not exist",
            )
        )


def test_prepare_documents_no_pdf(config, tmp_path, mock_documents):
    """Test failure when no linked documents could be found or created."""
    config.default_context_type = ContextType.FULL_DOCUMENT
    linked_doc_path = tmp_path / "linked_documents"
    # Don't create the directory
    pdf_dir = None

    with (
        patch("deet.extractors.cli_helpers.notify"),
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
        assert "no pdf dir supplied" in mock_fail.call_args[0][0]
