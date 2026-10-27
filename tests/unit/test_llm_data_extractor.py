"""Tests for the LLM data extractor module."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.decisions import (
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
)
from loguru import logger
from pydantic import ValidationError

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
    LLMInputSchema,
    build_llm_response_model,
)
from deet.data_models.documents import ContextType, Document
from deet.data_models.eppi import (
    EppiAttribute,
    EppiAttributeSelectionType,
    EppiDocument,
)
from deet.data_models.extraction import DocumentExtractionResult, ExtractionRunOutput
from deet.extractors.base_extractor import DataExtractionConfig, PromptConfig
from deet.extractors.llm_data_extractor import (
    LLMDataExtractor,
)
from deet.processors.csv_annotation_converter import CSVAnnotationConverter
from deet.settings import DecisionApiType, LLMProvider

# @pytest.fixture(autouse=True)
# def mock_settings(monkeypatch):
#     """
#     Fixture to mock the module-level settings object in llm_data_extractor.
#     This is necessary because the module loads settings at import time.
#     `autouse=True` ensures it runs for every test in this file.
#     """
#     mock_settings_obj = MagicMock()
#     mock_settings_obj.llm_model = "test-model"
#     mock_settings_obj.llm_temperature = 0.1
#     mock_settings_obj.llm_max_tokens = 1024
#     mock_settings_obj.llm_max_context_tokens = None
#     mock_settings_obj.azure_deployment = "test-deployment"
#     mock_settings_obj.azure_api_key.get_secret_value.return_value = "test-key"
#     mock_settings_obj.azure_api_base.get_secret_value.return_value = "test-base"
#     mock_settings_obj.llm_provider = "ollama"

#     monkeypatch.setattr(
#         "deet.extractors.llm_data_extractor.settings", mock_settings_obj
#     )
#     return mock_settings_obj


@pytest.fixture
def mock_settings() -> MagicMock:
    """
    Fixture providing a mock DataExtractionSettings instance.

    Settings are now injected explicitly into LLMDataExtractor rather than
    loaded at module import time, so this is passed directly to the
    constructor in tests instead of being patched onto the module.

    Note: no `spec=` is used here because the LLMDataExtractor constructor
    accesses several nested attributes (e.g. `azure_api_key.get_secret_value`)
    and a strict spec would reject any attribute not explicitly defined on
    the real settings model, causing spurious AttributeErrors.
    """
    mock_settings_obj = MagicMock()
    mock_settings_obj.azure_deployment = "test-deployment"
    mock_settings_obj.azure_api_key.get_secret_value.return_value = "test-key"
    mock_settings_obj.azure_api_base.get_secret_value.return_value = "test-base"
    mock_settings_obj.ollama_api_base = "http://localhost:11434"
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
        return LLMDataExtractor(config=default_config, settings=mock_settings)


def create_llm_extractor(default_config, mock_settings) -> LLMDataExtractor:
    """
    Create an LLMDataExtractor instance.

    Useful when we want to test different configuration options.
    """
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        return LLMDataExtractor(config=default_config, settings=mock_settings)


@pytest.fixture
def mock_litellm_completion():
    """Fixture to mock the litellm.completion call."""
    with patch("litellm.completion") as mock_completion:
        response_content = {
            "attribute_1234": {
                "output_data": True,
                "reasoning": "Found in text.",
                "additional_text": "Citation here.",
            },
            "attribute_2345": {
                "output_data": False,
                "reasoning": "Not found in text.",
                "additional_text": "No citation.",
            },
        }
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(response_content)
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 42
        mock_response.usage = mock_usage
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


def test_model_validator_populates_max_context_tokens(mock_settings):
    """Test that model_validator sets max_context_tokens from model when None."""
    with patch(
        "deet.extractors.base_extractor.get_model_max_tokens",
        return_value=128000,
    ):
        config = DataExtractionConfig()
    assert config.max_context_tokens == 128000


def test_model_validator_respects_user_override(mock_settings):
    """Test that model_validator does not override explicit max_context_tokens."""
    with patch(
        "deet.extractors.base_extractor.get_model_max_tokens",
        return_value=128000,
    ):
        config = DataExtractionConfig(max_context_tokens=5000)
    assert config.max_context_tokens == 5000


# core class
def test_llm_extractor_logs_max_tokens_when_set(mock_settings):
    """Test that LLMDataExtractor logs max_tokens when config has it set."""
    log_buf = StringIO()
    handler_id = logger.add(log_buf, format="{message}")
    try:
        config = DataExtractionConfig(max_tokens=512)
        with patch("pathlib.Path.read_text", return_value="Default system prompt"):
            LLMDataExtractor(config=config, settings=mock_settings)
        assert "max_tokens=512" in log_buf.getvalue()
    finally:
        logger.remove(handler_id)


def test_llm_extractor_init_custom_prompt(
    default_config, tmp_path: Path, mock_settings
):
    """Test LLMDataExtractor initialization with a custom system prompt."""
    custom_prompt_file = tmp_path / "custom.txt"
    custom_prompt_file.write_text("This is a custom prompt.")

    # path for default config
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        config = DataExtractionConfig()

    extractor = LLMDataExtractor(
        config=config,
        settings=mock_settings,
        custom_system_prompt_file=custom_prompt_file,
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


def test_call_llm_raises_when_payload_exceeds_max_by_default(
    llm_extractor, sample_eppi_attributes, mock_litellm_completion
):
    """Test _call_llm raises when payload exceeds max and truncation is off."""
    long_context = " ".join(["word"] * 2000)
    prompt = json.dumps(
        {"context": long_context, "attributes": []},
        ensure_ascii=False,
    )
    llm_extractor.config.max_context_tokens = 1000
    llm_extractor.config.truncate_on_overflow = False
    response_model = build_llm_response_model(sample_eppi_attributes)
    with pytest.raises(ValueError, match="exceeds max_context_tokens"):
        llm_extractor._call_llm(prompt=prompt, response_model=response_model)


def test_call_llm_truncates_when_truncate_on_overflow_enabled(
    llm_extractor, sample_eppi_attributes, mock_litellm_completion
):
    """Test that _call_llm truncates context when truncate_on_overflow is True."""
    long_context = " ".join(["word"] * 2000)
    prompt = json.dumps(
        {"context": long_context, "attributes": []},
        ensure_ascii=False,
    )
    llm_extractor.config.max_context_tokens = 1000
    llm_extractor.config.truncate_on_overflow = True
    response_model = build_llm_response_model(sample_eppi_attributes)
    llm_extractor._call_llm(prompt=prompt, response_model=response_model)
    call_args = mock_litellm_completion.call_args
    user_content = json.loads(call_args.kwargs["messages"][1]["content"])
    assert len(user_content["context"]) < len(long_context)


def test_call_llm_truncates_to_empty_when_system_and_attributes_exceed_max(
    llm_extractor, sample_eppi_attributes, mock_litellm_completion
):
    """Test that _call_llm sets context to empty when system+attributes exceed max."""
    long_context = " ".join(["word"] * 100)
    prompt = json.dumps(
        {"context": long_context, "attributes": []},
        ensure_ascii=False,
    )
    llm_extractor.config.max_context_tokens = 5
    llm_extractor.config.truncate_on_overflow = True
    response_model = build_llm_response_model(sample_eppi_attributes)
    llm_extractor._call_llm(prompt=prompt, response_model=response_model)
    call_args = mock_litellm_completion.call_args
    user_content = json.loads(call_args.kwargs["messages"][1]["content"])
    assert user_content["context"] == ""


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
def test_call_llm(
    mock_litellm_completion, mock_settings, llm_provider, sample_eppi_attributes
):
    """Test the _call_llm method."""
    prompt = '{"key": "value"}'
    response_model = build_llm_response_model(sample_eppi_attributes)

    if llm_provider in [member.value for member in LLMProvider]:
        config = DataExtractionConfig(model="test-model", provider=llm_provider)

        llm_extractor = create_llm_extractor(config, mock_settings)
        response, messages, output_tokens, input_tokens = llm_extractor._call_llm(
            prompt, response_model=response_model
        )

        mock_litellm_completion.assert_called_once()
        call_args = mock_litellm_completion.call_args
        if config.provider == LLMProvider.AZURE:
            assert call_args.kwargs["model"] == f"azure/{config.model}"
        elif config.provider == LLMProvider.OLLAMA:
            assert call_args.kwargs["model"] == f"ollama/{config.model}"
        else:
            assert config.provider in LLMProvider
        assert call_args.kwargs["response_format"]["type"] == "json_schema"
        assert (
            "llm_annotation_response"
            in call_args.kwargs["response_format"]["json_schema"]["name"]
        )
        assert response is not None
        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert output_tokens == 42
        assert input_tokens == 100
    else:
        with pytest.raises(ValidationError, match=r"provider"):
            DataExtractionConfig(model="test-model", provider=llm_provider)


def test_call_llm_forwards_ollama_num_ctx(
    mock_litellm_completion, mock_settings, sample_eppi_attributes
):
    """Ollama requests should pass num_ctx matching max_context_tokens."""
    mock_settings.llm_provider = LLMProvider.OLLAMA
    config = DataExtractionConfig(
        model="test-model",
        provider=LLMProvider.OLLAMA,
        max_context_tokens=16384,
    )
    llm_extractor = create_llm_extractor(config, mock_settings)
    response_model = build_llm_response_model(sample_eppi_attributes)

    llm_extractor._call_llm('{"key": "value"}', response_model=response_model)

    call_args = mock_litellm_completion.call_args
    assert call_args.kwargs["num_ctx"] == 16384


def test_parse_llm_response(
    llm_extractor, sample_eppi_attributes, sample_eppi_document
):
    """Test successful parsing of a valid keyed LLM response."""
    response_model = build_llm_response_model(sample_eppi_attributes)
    response_content = json.dumps(
        {
            "attribute_1234": {
                "output_data": True,
                "reasoning": "Found.",
                "additional_text": "Citation.",
            },
            "attribute_2345": {
                "output_data": False,
                "reasoning": "Not found.",
                "additional_text": "No citation.",
            },
        }
    )
    annotations = llm_extractor._parse_llm_response(
        response_content, response_model, sample_eppi_attributes
    )
    # The dynamic schema requires a response for every selected attribute.
    assert len(annotations) == 2
    by_id = {a.attribute.attribute_id: a for a in annotations}
    assert all(isinstance(a, GoldStandardAnnotation) for a in annotations)
    assert by_id[1234].output_data is True
    assert by_id[2345].output_data is False
    assert by_id[1234].annotation_type == AnnotationType.LLM


def test_parse_llm_response_missing_attribute_raises(
    llm_extractor,
    sample_eppi_attributes,
):
    """A response omitting a required attribute must fail validation."""
    response_model = build_llm_response_model(sample_eppi_attributes)
    # Only one of the two required attributes is present.
    incomplete_response = json.dumps(
        {
            "attribute_1234": {
                "output_data": True,
                "reasoning": "Found.",
                "additional_text": "Citation.",
            }
        }
    )
    with pytest.raises(ValidationError):
        llm_extractor._parse_llm_response(
            incomplete_response, response_model, sample_eppi_attributes
        )


def test_parse_llm_response_attribute_lookup_mismatch_raises(
    llm_extractor,
    sample_eppi_attributes,
):
    """A validated model field with no matching attribute must raise."""
    response_model = build_llm_response_model(sample_eppi_attributes)
    valid_response = json.dumps(
        {
            "attribute_1234": {
                "output_data": True,
                "reasoning": "Found.",
                "additional_text": "Citation.",
            },
            "attribute_2345": {
                "output_data": False,
                "reasoning": "Not found.",
                "additional_text": "No citation.",
            },
        }
    )
    mismatched_attributes = [sample_eppi_attributes[0]]
    with pytest.raises(ValueError, match="No attribute found for ID: 2345"):
        llm_extractor._parse_llm_response(
            valid_response, response_model, mismatched_attributes
        )


def test_parse_llm_response_validation_error(
    llm_extractor,
    sample_eppi_attributes,
):
    """Test that _parse_llm_response raises ValidationError for bad schema."""
    response_model = build_llm_response_model(sample_eppi_attributes)
    invalid_response = json.dumps(
        {
            "attribute_1234": {"output_data": True},
            "attribute_2345": {"output_data": False},
        }
    )  # Missing additional_text/reasoning fields
    with pytest.raises(ValidationError):
        llm_extractor._parse_llm_response(
            invalid_response, response_model, sample_eppi_attributes
        )


def test_parse_llm_response_json_decode_error(
    llm_extractor,
    sample_eppi_attributes,
):
    """Test that _parse_llm_response raises ValueError for invalid JSON."""
    response_model = build_llm_response_model(sample_eppi_attributes)
    invalid_json = "this is not json"
    with pytest.raises(ValueError, match="Invalid JSON"):
        llm_extractor._parse_llm_response(
            invalid_json, response_model, sample_eppi_attributes
        )


def test_extract_from_document(
    llm_extractor,
    sample_eppi_document,
    sample_eppi_attributes,
    mock_litellm_completion,
):
    """Test the end-to-end flow of extract_from_document."""
    payload = "This is the full text of the document."
    result = llm_extractor.extract_from_document(
        sample_eppi_attributes,
        payload=payload,
        context_type=ContextType.FULL_DOCUMENT,
    )
    assert isinstance(result, DocumentExtractionResult)
    # The dynamic schema requires a response for every selected attribute.
    assert len(result.annotations) == 2
    assert isinstance(result.messages, list)
    assert len(result.messages) >= 1
    assert result.output_tokens == 42
    assert result.input_tokens == 100
    assert result.model is not None
    assert result.timestamp is not None
    assert {a.attribute.attribute_id for a in result.annotations} == {1234, 2345}
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
    """Test extracting from multiple documents (document mode)."""
    sample_eppi_documents = [sample_eppi_document]
    llm_extractor.config.default_context_type = ContextType.ABSTRACT_ONLY
    output_file = tmp_path / "results.json"
    run_output = llm_extractor.extract_from_documents(
        attributes=sample_eppi_attributes,
        documents=sample_eppi_documents,
        output_file=output_file,
    )
    assert isinstance(run_output, ExtractionRunOutput)
    assert len(run_output.annotated_documents) == 1
    assert mock_litellm_completion.call_count == 1

    saved = json.loads(output_file.read_text())
    assert "annotated_documents" in saved
    assert "metadata" in saved
    assert len(saved["annotated_documents"]) == 1

    meta = saved["metadata"]
    assert meta["total_input_tokens"] == 100
    assert meta["total_output_tokens"] == 42
    doc_id_str = str(sample_eppi_document.safe_identity.document_id)
    assert meta["per_document"][doc_id_str]["input_tokens"] == 100
    assert meta["per_document"][doc_id_str]["output_tokens"] == 42
    assert "total_cost_usd" in meta


def test_extract_from_documents_skips_document_missing_abstract(
    llm_extractor,
    mock_litellm_completion,
):
    """
    Test that a document with no abstract is skipped (not raised) under
    ABSTRACT_ONLY context, a warning is logged, and other documents still
    process normally.
    """
    # The dynamic schema requires a response for every selected attribute, so
    # both ids here must match the keys in mock_litellm_completion's response.
    attributes = [
        Attribute(
            attribute_id=1234,
            attribute_label="Attribute 1",
            output_data_type=AttributeType.BOOL,
        ),
        Attribute(
            attribute_id=2345,
            attribute_label="Attribute 2",
            output_data_type=AttributeType.BOOL,
        ),
    ]

    converter = CSVAnnotationConverter(base_output_dir=None)
    mapping = {"abstract": "abstract"}
    doc_with_abstract = Document(
        name="Has Abstract",
        document_id=1,
        citation=converter.build_destiny_reference(
            {"abstract": "The abstract."}, mapping
        ),
    )
    doc_without_abstract = Document(
        name="No Abstract",
        document_id=2,
        citation=converter.build_destiny_reference({}, mapping={}),
    )

    messages = []
    logger_id = logger.add(messages.append, level="WARNING")
    run_output = llm_extractor.extract_from_documents(
        attributes=attributes,
        documents=[doc_without_abstract, doc_with_abstract],
        context_type=ContextType.ABSTRACT_ONLY,
    )
    logger.remove(logger_id)

    assert len(run_output.annotated_documents) == 1
    assert run_output.annotated_documents[0].document.name == "Has Abstract"
    assert any("No abstract found" in str(m) for m in messages)


def test_extract_from_documents_continues_on_error(
    llm_extractor,
    sample_eppi_document,
    sample_eppi_attributes,
    mock_litellm_completion,
    tmp_path,
):
    """Test that when one document fails, processing continues with empty results."""
    sample_eppi_documents = [sample_eppi_document]
    llm_extractor.config.default_context_type = ContextType.ABSTRACT_ONLY
    mock_litellm_completion.side_effect = ValueError("LLM call failed")
    run_output = llm_extractor.extract_from_documents(
        attributes=sample_eppi_attributes,
        documents=sample_eppi_documents,
    )
    assert isinstance(run_output, ExtractionRunOutput)
    assert run_output.annotated_documents == []
    assert run_output.metadata.total_input_tokens == 0
    assert run_output.metadata.total_output_tokens == 0
    assert mock_litellm_completion.call_count == 1


def make_decision_response(
    answers: list, input_tokens: int = 90, output_tokens: int = 0
) -> OpenAIDecisionResponse:
    """Build a litellm decisions response to serve from mocks."""
    return OpenAIDecisionResponse(
        model="s1-dep",
        answers=tuple(answers),
        usage=OpenAIDecisionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


def make_completion_response(content: dict) -> MagicMock:
    """Build a litellm completion response to serve from mocks."""
    response = MagicMock()
    response.choices[0].message.content = json.dumps(content)
    response.usage.prompt_tokens = 80
    response.usage.completion_tokens = 20
    return response


def create_decision_extractor(
    mock_settings: MagicMock,
    decision_api_type: DecisionApiType = DecisionApiType.OPENAI,
    decision_threshold: float = 0.5,
    decision_template: str | None = None,
) -> LLMDataExtractor:
    """Build an Azure-provider extractor with decision routing configured."""
    with patch("pathlib.Path.read_text", return_value="Default system prompt"):
        config = DataExtractionConfig(
            model="test-model",
            provider=LLMProvider.AZURE,
            decision_model="s1-dep",
            decision_api_type=decision_api_type,
            decision_threshold=decision_threshold,
        )
        extractor = LLMDataExtractor(config=config, settings=mock_settings)
    if decision_template is not None:
        extractor.config.prompt_config.decision_question_template = decision_template
    return extractor


@pytest.fixture
def mock_litellm_decisions():
    """Fixture to mock the litellm.decisions call for both sample bool attrs."""
    with patch("litellm.decisions") as mock_decisions:
        mock_decisions.return_value = make_decision_response(
            [
                OpenAIPredicateAnswer(name="attribute_1234", probability=0.93),
                OpenAIPredicateAnswer(name="attribute_2345", probability=0.12),
            ]
        )
        yield mock_decisions


@pytest.fixture
def sample_bool_attrs(make_attr) -> list[Attribute]:
    """Two bool attributes matching the names the canned mocks answer."""
    return [make_attr(1234), make_attr(2345, "What is the question?")]


class TestDecisionRouteConstruction:
    """__init__ turns decision config into the model prefix and api_base."""

    def test_openai_route_builds_prefix_and_base(self, mock_settings):
        """openai mode prefixes the deployment and mounts the /openai route."""
        extractor = create_decision_extractor(mock_settings)
        assert extractor.decision_model == "openai/s1-dep"
        assert extractor.decision_api_base == "test-base/openai"

    def test_system_one_route_builds_prefix_and_base(self, mock_settings):
        """system_one mode uses azure_ai at the bare api_base."""
        extractor = create_decision_extractor(
            mock_settings, decision_api_type=DecisionApiType.SYSTEM_ONE
        )
        assert extractor.decision_model == "azure_ai/s1-dep"
        assert extractor.decision_api_base == "test-base"

    def test_decision_model_requires_azure_provider(self, mock_settings):
        """Configuring a decision model on a non-Azure provider is refused."""
        with pytest.raises(
            ValueError, match="decision_model requires the Azure provider"
        ):
            create_llm_extractor(
                DataExtractionConfig(
                    model="test-model",
                    provider=LLMProvider.OLLAMA,
                    decision_model="s1-dep",
                ),
                mock_settings,
            )


class TestAnnotateRouting:
    """_annotate splits attributes by type across the two endpoints."""

    def test_without_decision_model_uses_completion_only(
        self,
        llm_extractor,
        mock_litellm_decisions,
        mock_litellm_completion,
        sample_bool_attrs,
    ):
        """With decision_model unset, every attribute goes to completions."""
        calls = llm_extractor._annotate(
            context="doc text", attributes=sample_bool_attrs
        )
        assert len(calls) == 1
        mock_litellm_decisions.assert_not_called()
        mock_litellm_completion.assert_called_once()

    def test_all_bool_calls_decision_only(
        self, mock_settings, mock_litellm_decisions, sample_bool_attrs
    ):
        """All-boolean attribute sets never hit the completion endpoint."""
        with patch("litellm.completion") as mock_completion:
            extractor = create_decision_extractor(mock_settings)
            calls = extractor._annotate(
                context="doc text", attributes=sample_bool_attrs
            )
        mock_completion.assert_not_called()
        mock_litellm_decisions.assert_called_once()
        kwargs = mock_litellm_decisions.call_args.kwargs
        assert kwargs["model"] == "openai/s1-dep"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["api_base"] == "test-base/openai"
        assert kwargs["input"] == "doc text"
        assert [q["name"] for q in kwargs["questions"]] == [
            "attribute_1234",
            "attribute_2345",
        ]
        result = DocumentExtractionResult.combine(calls)
        assert {a.attribute.attribute_id for a in result.annotations} == {1234, 2345}

    def test_none_bool_calls_completion_only(
        self, mock_settings, mock_litellm_decisions, make_attr
    ):
        """Non-boolean attribute sets never hit the decision endpoint."""
        string_attr = make_attr(
            3456, "Summarise the method.", output_data_type=AttributeType.STRING
        )
        completion_payload = {
            "attribute_3456": {
                "output_data": "A stated method.",
                "reasoning": "r",
                "additional_text": "",
            }
        }
        with patch(
            "litellm.completion",
            return_value=make_completion_response(completion_payload),
        ) as mock_completion:
            extractor = create_decision_extractor(mock_settings)
            calls = extractor._annotate(context="doc text", attributes=[string_attr])
        mock_litellm_decisions.assert_not_called()
        mock_completion.assert_called_once()
        assert len(calls) == 1
        assert calls[0].model == "azure/test-model"

    def test_mixed_calls_both_endpoints_and_merges(
        self,
        mock_settings,
        mock_litellm_decisions,
        sample_bool_attrs,
        make_attr,
    ):
        """Mixed sets split across endpoints and merge into one combined result."""
        string_attr = make_attr(
            3456, "Summarise the method.", output_data_type=AttributeType.STRING
        )
        completion_payload = {
            "attribute_3456": {
                "output_data": "A stated method.",
                "reasoning": "cited",
                "additional_text": "",
            }
        }
        with patch(
            "litellm.completion",
            return_value=make_completion_response(completion_payload),
        ) as mock_completion:
            extractor = create_decision_extractor(mock_settings)
            calls = extractor._annotate(
                context="doc text",
                attributes=[*sample_bool_attrs, string_attr],
            )
        mock_litellm_decisions.assert_called_once()
        mock_completion.assert_called_once()
        names = [
            q["name"] for q in mock_litellm_decisions.call_args.kwargs["questions"]
        ]
        assert names == ["attribute_1234", "attribute_2345"]
        result = DocumentExtractionResult.combine(calls)
        assert {a.attribute.attribute_id for a in result.annotations} == {
            1234,
            2345,
            3456,
        }
        assert result.model == "openai/s1-dep + azure/test-model"


class TestCallDecision:
    """_call_decision sends the questions and assembles the extraction result."""

    def test_raises_on_non_bool_attribute(self, mock_settings, make_attr):
        """_call_decision is strict about BOOL-only attributes."""
        string_attr = make_attr(
            3456, "Summarise the method.", output_data_type=AttributeType.STRING
        )
        extractor = create_decision_extractor(mock_settings)
        with pytest.raises(ValueError, match="only supports BOOL"):
            extractor._call_decision(context="doc text", attributes=[string_attr])

    def test_assembles_result(
        self, mock_settings, mock_litellm_decisions, sample_bool_attrs
    ):
        """The result carries annotations, messages, tokens and model."""
        extractor = create_decision_extractor(mock_settings)
        result = extractor._call_decision(
            context="doc text", attributes=sample_bool_attrs
        )
        assert [a.raw_data for a in result.annotations] == [True, False]
        assert result.input_tokens == 90
        assert result.output_tokens == 0
        assert result.model == "openai/s1-dep"
        assert result.messages[0]["role"] == "system"
        assert "decisions_questions" in result.messages[0]["content"]
        assert result.messages[1] == {"role": "user", "content": "doc text"}


class TestParseDecisionResponse:
    """_parse_decision_response turns answers into thresholded annotations."""

    @staticmethod
    def parse(
        extractor: LLMDataExtractor, answers: list, attributes: list[Attribute]
    ) -> tuple[list[GoldStandardAnnotation], list[str]]:
        """Parse a response built from answers, capturing warnings."""
        messages: list[str] = []
        logger_id = logger.add(messages.append, level="WARNING")
        try:
            annotations = extractor._parse_decision_response(
                response=make_decision_response(answers), attributes=attributes
            )
        finally:
            logger.remove(logger_id)
        return annotations, messages

    def test_thresholds_probability_into_raw_data(
        self, mock_settings, sample_bool_attrs
    ):
        """raw_data holds the thresholded bool, reasoning the raw probability."""
        extractor = create_decision_extractor(mock_settings)
        annotations, messages = self.parse(
            extractor,
            [
                OpenAIPredicateAnswer(name="attribute_1234", probability=0.93),
                OpenAIPredicateAnswer(name="attribute_2345", probability=0.12),
            ],
            sample_bool_attrs,
        )
        assert [a.raw_data for a in annotations] == [True, False]
        assert [a.reasoning for a in annotations] == ["0.9300", "0.1200"]
        assert [a.annotation_type for a in annotations] == [
            AnnotationType.LLM,
            AnnotationType.LLM,
        ]
        assert messages == []

    def test_threshold_is_inclusive_and_configurable(
        self, mock_settings, sample_bool_attrs
    ):
        """probability >= decision_threshold yields True, at or above the set value."""
        extractor = create_decision_extractor(mock_settings, decision_threshold=0.3)
        annotations, _ = self.parse(
            extractor,
            [
                OpenAIPredicateAnswer(name="attribute_1234", probability=0.3),
                OpenAIPredicateAnswer(name="attribute_2345", probability=0.3),
            ],
            sample_bool_attrs,
        )
        assert [a.raw_data for a in annotations] == [True, True]

    def test_refusal_skips_attribute(self, mock_settings, sample_bool_attrs):
        """A non-predicate (refusal) answer produces no annotation for that attr."""
        extractor = create_decision_extractor(mock_settings)
        annotations, messages = self.parse(
            extractor,
            [
                OpenAIPredicateAnswer(name="attribute_1234", probability=0.8),
                OpenAIRefusalAnswer(name="attribute_2345"),
            ],
            sample_bool_attrs,
        )
        assert [a.attribute.attribute_id for a in annotations] == [1234]
        assert len(messages) == 1
        assert "no predicate answer for 1 of 2" in str(messages[0])

    def test_unexpected_answer_names_warn(self, mock_settings, sample_bool_attrs):
        """Answers under names that match no attribute are reported."""
        extractor = create_decision_extractor(mock_settings)
        annotations, messages = self.parse(
            extractor,
            [
                OpenAIPredicateAnswer(name="attribute_1234", probability=0.8),
                OpenAIPredicateAnswer(name="renamed_2345", probability=0.8),
            ],
            sample_bool_attrs,
        )
        assert [a.attribute.attribute_id for a in annotations] == [1234]
        assert any("['attribute_2345']" in str(m) for m in messages)
        assert any("['renamed_2345']" in str(m) for m in messages)


class TestDecisionInstructions:
    """_decision_instructions frames attribute prompts as predicate questions."""

    def test_pass_through_question_prompts(self, mock_settings, make_attr):
        """Prompts already ending in '?' are sent unchanged."""
        extractor = create_decision_extractor(mock_settings)
        attribute = make_attr(2345, "What is the question?")
        assert extractor._decision_instructions(attribute) == "What is the question?"

    def test_apply_default_template(self, mock_settings, make_attr):
        """Non-question prompts are framed by the default template."""
        attribute = make_attr(4567, "Heat waves as an adaptation measure")
        extractor = create_decision_extractor(mock_settings)
        assert extractor._decision_instructions(attribute) == (
            "Does this document concern Attribute 4567?"
            " Heat waves as an adaptation measure"
        )

    def test_raw_template_passes_prompt_through(self, mock_settings, make_attr):
        """A '$prompt' template sends the attribute prompt as-is."""
        attribute = make_attr(4567, "Heat waves as an adaptation measure")
        extractor = create_decision_extractor(
            mock_settings, decision_template="$prompt"
        )
        assert (
            extractor._decision_instructions(attribute)
            == "Heat waves as an adaptation measure"
        )


class TestDecisionQuestionTemplateValidation:
    """PromptConfig rejects templates that would fail at substitution time."""

    @pytest.mark.parametrize(
        "template",
        ["$prompt", "Is $label relevant? $prompt", "Costs $$5? $prompt"],
    )
    def test_accepts_valid_templates(self, template):
        """Templates using only $label, $prompt and escaped '$$' are accepted."""
        config = PromptConfig(system_prompt="sys", decision_question_template=template)
        assert config.decision_question_template == template

    @pytest.mark.parametrize(
        "template",
        ["Does it concern $topic?", "Costs $5? $prompt", "Trailing $"],
    )
    def test_rejects_invalid_templates(self, template):
        """Unknown placeholders and stray '$' fail at config load."""
        with pytest.raises(ValidationError, match="decision_question_template"):
            PromptConfig(system_prompt="sys", decision_question_template=template)
