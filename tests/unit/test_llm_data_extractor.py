"""Tests for the LLM data extractor module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from deet.data_models.base import AnnotationType, GoldStandardAnnotation, LLMInputSchema
from deet.data_models.documents import ContextType
from deet.data_models.eppi import (
    AttributeType,
    EppiAttribute,
    EppiAttributeSelectionType,
    EppiDocument,
)
from deet.extractors.llm_data_extractor import (
    DataExtractionConfig,
    LLMDataExtractor,
    PromptConfig,
)
from deet.settings import LLMProvider


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Fixture to mock the module-level settings object in llm_data_extractor.
    This is necessary because the module loads settings at import time.
    `autouse=True` ensures it runs for every test in this file.
    """
    mock_settings_obj = MagicMock()
    mock_settings_obj.llm_model = "test-model"
    mock_settings_obj.llm_temperature = 0.1
    mock_settings_obj.llm_max_tokens = 1024
    mock_settings_obj.azure_deployment = "test-deployment"
    mock_settings_obj.azure_api_key.get_secret_value.return_value = "test-key"
    mock_settings_obj.azure_api_base.get_secret_value.return_value = "test-base"
    mock_settings_obj.llm_provider = "ollama"

    monkeypatch.setattr(
        "deet.extractors.llm_data_extractor.settings", mock_settings_obj
    )
    return mock_settings_obj


@pytest.fixture
def sample_eppi_document() -> EppiDocument:
    """Fixture for a sample EppiDocument."""
    reference = {
        "document_id": 12345,
        "name": "Test Document",
        "context": "This is the abstract.",
        "Abstract": "The document's abstract.",
    }
    return EppiDocument.model_validate(reference)


@pytest.fixture
def sample_eppi_attributes() -> list[EppiAttribute]:
    """Fixture for a list of sample EppiAttributes."""
    return [
        EppiAttribute(  # type: ignore [call-arg] # old
            attribute_id=1234,
            attribute_label="Attribute 1",
            output_data_type=AttributeType.BOOL,
            attribute_set_description="Is attribute 1 present?",
            attribute_type=EppiAttributeSelectionType.SELECTABLE,
        ),
        EppiAttribute(  # type: ignore [call-arg]# new
            attribute_id=2345,
            prompt="What is the question?",
            attribute_label="Attribute 2",
            output_data_type=AttributeType.BOOL,
            attribute_set_description="foo",
            attribute_type=EppiAttributeSelectionType.SELECTABLE,
        ),
    ]


@pytest.fixture
def default_config() -> DataExtractionConfig:
    """Fixture for a default DataExtractionConfig."""
    return DataExtractionConfig()


@pytest.fixture
def llm_extractor(request, default_config, mock_settings) -> LLMDataExtractor:
    """Fixture for an LLMDataExtractor instance."""
    # Patch file reads in the PromptConfig validator
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        return LLMDataExtractor(config=default_config)


def create_llm_extractor(default_config, mock_settings) -> LLMDataExtractor:
    """
    Create an LLMDataExtractor instance.

    Useful when we want to test different configuration options.
    """
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        return LLMDataExtractor(config=default_config)


@pytest.fixture
def mock_litellm_completion():
    """Fixture to mock the litellm.completion call."""
    with patch("litellm.completion") as mock_completion:
        response_content = {
            "annotations": [
                {
                    "attribute_id": 1234,
                    "output_data": True,
                    "reasoning": "Found in text.",
                    "additional_text": "Citation here.",
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(response_content)
        mock_completion.return_value = mock_response
        yield mock_completion


# config
def test_prompt_config_load_from_file(tmp_path: Path):
    """Test that PromptConfig loads the system prompt from a file."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Custom system prompt")
    config = PromptConfig(system_prompt=prompt_file)
    assert config.system_prompt == "Custom system prompt"


def test_prompt_config_missing_file(tmp_path: Path):
    """Test that PromptConfig raises ValueError for a missing prompt file."""
    with pytest.raises(ValueError, match="not found"):
        PromptConfig(system_prompt=tmp_path / "nonexistent.txt")


# core class
def test_llm_extractor_init_custom_prompt(default_config, tmp_path: Path):
    """Test LLMDataExtractor initialization with a custom system prompt."""
    custom_prompt_file = tmp_path / "custom.txt"
    custom_prompt_file.write_text("This is a custom prompt.")

    # path for default config
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        config = DataExtractionConfig()

    extractor = LLMDataExtractor(
        config=config, custom_system_prompt_file=custom_prompt_file
    )
    assert extractor.config.prompt_config.system_prompt == "This is a custom prompt."


def test_filter_attributes(llm_extractor, sample_eppi_attributes):
    """Test the _filter_attributes method."""
    filter_ids = [1234]
    filtered = llm_extractor._filter_attributes(
        sample_eppi_attributes, filter_ids=filter_ids
    )
    assert len(filtered) == 1
    assert filtered[0].attribute_id == 1234


def test_filter_attributes_no_selection(llm_extractor, sample_eppi_attributes):
    """Test _filter_attributes when no IDs are selected."""
    filtered = llm_extractor._filter_attributes(sample_eppi_attributes, filter_ids=None)
    assert len(filtered) == 2


@pytest.mark.parametrize(
    "filter_ids",
    [
        ["bad_id_1", "bad_id_2", 12345, 6789],
        ["bad_id_1", "bad_id_2"],
        [{"test_key": "test_value"}, [1, 2, 3]],
    ],
)
def test_extract_from_document_bad_filter_list(
    llm_extractor, sample_eppi_document, sample_eppi_attributes, filter_ids
):
    """
    Test extract_from_document raises ValueError if the filter list cannot
    entirely be cast to integers.
    """
    payload = "This is the full text of the document."
    with pytest.raises(ValueError, match="No attributes selected"):
        llm_extractor.extract_from_document(
            attributes=sample_eppi_attributes,
            payload=payload,
            filter_attribute_ids=filter_ids,
            context_type=ContextType.FULL_DOCUMENT,
        )


def test_prepare_context_full_document_context_in_config(
    llm_extractor, sample_eppi_document
):
    """Test _prepare_context with FULL_DOCUMENT type."""
    payload = "This is the full text."
    llm_extractor.config.default_context_type = ContextType.FULL_DOCUMENT
    context = llm_extractor._prepare_context(payload=payload)
    assert context == payload


def test_prepare_context_abstract_only(llm_extractor, sample_eppi_document):
    """Test _prepare_context with ABSTRACT_ONLY type."""
    llm_extractor.config.default_context_type = ContextType.ABSTRACT_ONLY
    sample_eppi_document.set_abstract_context()
    assert sample_eppi_document.context == sample_eppi_document.abstract


def test_prepare_context_truncation(llm_extractor, sample_eppi_document):
    """Test that context is truncated if it exceeds max length."""
    payload = "This is the very long full text of the document."
    llm_extractor.config.max_context_length = 10
    context = llm_extractor._prepare_context(payload=payload)
    assert len(context) <= 13  # 10 chars + "..."
    assert context.endswith("...")


def test_prepare_context_not_implemented(
    llm_extractor: LLMDataExtractor, sample_eppi_document
):
    """Test that RAG and CUSTOM context types raise NotImplementedError."""
    payload = "This is the full text."
    llm_extractor.config.default_context_type = ContextType.RAG_SNIPPETS
    with pytest.raises(NotImplementedError):
        llm_extractor._prepare_context(payload=payload)


def test_generate_user_message_json(llm_extractor, sample_eppi_attributes):
    """Test the generation of the structured JSON user message."""
    context = "Sample context"
    json_str = llm_extractor._generate_user_message_json(
        context, sample_eppi_attributes
    )
    payload = json.loads(json_str)
    assert "context" in payload
    assert "attributes" in payload
    assert len(payload.keys()) == 2

    one_input_item = LLMInputSchema(**payload["attributes"][0])
    second_input_item = LLMInputSchema(**payload["attributes"][1])
    assert isinstance(one_input_item, LLMInputSchema)

    # we didn't tell it what to use as prompt, so use
    # whatever's in `attribute_label`
    assert one_input_item.prompt == "Attribute 1"
    assert one_input_item.prompt == sample_eppi_attributes[0].attribute_label

    # for att 2, we gave it a prompt
    assert second_input_item.prompt == "What is the question?"
    assert second_input_item.prompt != sample_eppi_attributes[1].attribute_label


@pytest.mark.parametrize("llm_provider", [*list(LLMProvider), "unsupported_provider"])
def test_call_llm(mock_litellm_completion, mock_settings, llm_provider):
    """Test the _call_llm method."""
    prompt = '{"key": "value"}'
    mock_settings.llm_provider = llm_provider

    if llm_provider in [member.value for member in LLMProvider]:
        config = DataExtractionConfig(
            model=mock_settings.llm_model, provider=mock_settings.llm_provider
        )

        llm_extractor = create_llm_extractor(config, mock_settings)
        response, messages = llm_extractor._call_llm(prompt)

        mock_litellm_completion.assert_called_once()
        call_args = mock_litellm_completion.call_args
        if mock_settings.llm_provider == LLMProvider.AZURE:
            assert call_args.kwargs["model"] == f"azure/{config.model}"
        elif mock_settings.llm_provider == LLMProvider.OLLAMA:
            assert call_args.kwargs["model"] == f"ollama/{config.model}"
        else:
            assert mock_settings.llm_provider in LLMProvider
        assert call_args.kwargs["response_format"]["type"] == "json_schema"
        assert (
            "llm_annotation_response"
            in call_args.kwargs["response_format"]["json_schema"]["name"]
        )
        assert response is not None
        assert isinstance(messages, list)
        assert len(messages) >= 1
    else:
        with pytest.raises(ValidationError, match=r"provider"):
            config = DataExtractionConfig(
                model=mock_settings.llm_model, provider=mock_settings.llm_provider
            )


def test_parse_llm_response(
    llm_extractor, sample_eppi_attributes, sample_eppi_document
):
    """Test successful parsing of a valid LLM response."""
    response_content = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": 1234,
                    "output_data": True,
                    "reasoning": "Found.",
                    "additional_text": "Citation.",
                }
            ]
        }
    )
    annotations = llm_extractor._parse_llm_response(
        response_content, sample_eppi_attributes
    )
    # it filters through ids which exist in both,
    # so even though the length of sample_eppi_attributes is
    # longer than response_content,
    # below is expeceted behaviour and we're happy.
    assert len(annotations) == 1
    annotation = annotations[0]
    assert isinstance(annotation, GoldStandardAnnotation)
    assert annotation.attribute.attribute_id == 1234
    assert annotation.output_data is True
    assert annotation.annotation_type == AnnotationType.LLM


def test_parse_llm_response_validation_error(
    llm_extractor,
    sample_eppi_attributes,
):
    """Test that _parse_llm_response raises ValidationError for bad schema."""
    invalid_response = json.dumps(
        {"annotations": [{"attribute_id": "attr1"}]}
    )  # Missing fields
    with pytest.raises(ValidationError):
        llm_extractor._parse_llm_response(invalid_response, sample_eppi_attributes)


def test_parse_llm_response_json_decode_error(
    llm_extractor,
    sample_eppi_attributes,
):
    """Test that _parse_llm_response raises ValueError for invalid JSON."""
    invalid_json = "this is not json"
    with pytest.raises(ValueError, match="Invalid JSON"):
        llm_extractor._parse_llm_response(invalid_json, sample_eppi_attributes)


def test_extract_from_document(
    llm_extractor,
    sample_eppi_document,
    sample_eppi_attributes,
    mock_litellm_completion,
):
    """Test the end-to-end flow of extract_from_document."""
    payload = "This is the full text of the document."
    annotations, messages = llm_extractor.extract_from_document(
        sample_eppi_attributes,
        payload=payload,
        context_type=ContextType.FULL_DOCUMENT,
    )
    assert len(annotations) == 1
    assert isinstance(messages, list)
    assert len(messages) >= 1
    assert annotations[0].attribute.attribute_id == 1234
    mock_litellm_completion.assert_called_once()


def test_extract_from_document_no_attributes(
    llm_extractor, sample_eppi_document, sample_eppi_attributes
):
    """Test extract_from_document raises ValueError if no attributes are selected."""
    payload = "This is the full text of the document."
    with pytest.raises(ValueError, match="No attributes selected"):
        llm_extractor.extract_from_document(
            sample_eppi_attributes,
            filter_attribute_ids=[999999],
            payload=payload,
            context_type=ContextType.FULL_DOCUMENT,
        )


def test_extract_from_documents(
    llm_extractor,
    sample_eppi_document,
    sample_eppi_attributes,
    mock_litellm_completion,
    tmp_path,
):
    """Test extracting from multiple documents (directory mode)."""
    sample_eppi_documents = [sample_eppi_document]
    llm_extractor.config.default_context_type = ContextType.ABSTRACT_ONLY
    all_annotations = llm_extractor.extract_from_documents(
        attributes=sample_eppi_attributes,
        documents=sample_eppi_documents,
    )
    assert len(all_annotations) == 1
    assert mock_litellm_completion.call_count == 1


def test_extract_from_documents_continues_on_error(
    llm_extractor,
    sample_eppi_document,
    sample_eppi_attributes,
    mock_litellm_completion,
    tmp_path,
):
    """Test that when one file fails, processing continues and returns empty dict."""
    sample_eppi_documents = [sample_eppi_document]
    llm_extractor.config.default_context_type = ContextType.ABSTRACT_ONLY
    mock_litellm_completion.side_effect = ValueError("LLM call failed")
    all_annotations = llm_extractor.extract_from_documents(
        attributes=sample_eppi_attributes,
        documents=sample_eppi_documents,
    )
    assert all_annotations == []
    assert mock_litellm_completion.call_count == 1
