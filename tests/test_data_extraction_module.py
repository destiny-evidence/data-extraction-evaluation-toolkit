"""Tests for the generalizable data extraction module."""

import json
from unittest.mock import patch
from uuid import uuid4

import pytest

from app.extractors.data_extraction_module import (
    ContextType,
    DataExtractionConfig,
    DataExtractionModule,
    PromptConfig,
    extract_all_attributes,
    extract_batch_attributes,
    extract_single_attribute,
)
from app.models.base import AnnotationType
from app.models.eppi import EppiAttribute, EppiDocument


def test_data_extraction_config_defaults() -> None:
    """Test DataExtractionConfig with default values."""
    config = DataExtractionConfig()

    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.1
    assert config.context_type == ContextType.FULL_DOCUMENT
    assert config.include_reasoning is True
    assert config.include_additional_text is True


def test_data_extraction_config_custom() -> None:
    """Test DataExtractionConfig with custom values."""
    prompt_config = PromptConfig(
        system_prompt="Custom system prompt",
        attribute_specific_prompt="Custom attribute prompt",
    )

    config = DataExtractionConfig(
        model="gpt-4",
        temperature=0.5,
        context_type=ContextType.ABSTRACT_ONLY,
        selected_attribute_ids=["123"],
        prompt_config=prompt_config,
        include_reasoning=False,
    )

    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    assert config.context_type == ContextType.ABSTRACT_ONLY
    assert config.selected_attribute_ids == ["123"]
    assert config.prompt_config.system_prompt == "Custom system prompt"
    assert config.include_reasoning is False


def test_data_extraction_module_initialization() -> None:
    """Test DataExtractionModule initialization."""
    with patch("os.getenv") as mock_getenv:
        mock_getenv.return_value = None  # No Azure environment variables
        config = DataExtractionConfig(model="gpt-4")
        module = DataExtractionModule(config)

        assert module.config.model == "gpt-4"
        assert module.model == "gpt-4"  # Should use config model when no Azure


def test_data_extraction_module_azure_model() -> None:
    """Test DataExtractionModule with Azure configuration."""
    with patch("os.getenv") as mock_getenv:

        def mock_getenv_side_effect(key: str, default: str | None = None) -> str | None:
            if key == "AZURE_API_KEY":
                return "test"
            if key == "AZURE_DEPLOYMENT":
                return "gpt-4"
            return default

        mock_getenv.side_effect = mock_getenv_side_effect
        config = DataExtractionConfig(model="gpt-4o-mini")
        module = DataExtractionModule(config)

        assert module.model == "azure/gpt-4"


def test_filter_attributes_no_ids_returns_all() -> None:
    """When no selected_attribute_ids are set, return all attributes."""
    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
    ]

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    filtered = module._filter_attributes(attributes)
    assert len(filtered) == 2
    assert filtered == attributes


def test_filter_attributes_selected_ids() -> None:
    """Filter attributes using selected_attribute_ids."""
    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
    ]

    config = DataExtractionConfig(selected_attribute_ids=["2"])
    module = DataExtractionModule(config)

    filtered = module._filter_attributes(attributes)
    assert len(filtered) == 1
    assert filtered[0].attribute_id == "2"


def test_filter_attributes_multiple_ids() -> None:
    """Filter multiple attributes using selected_attribute_ids."""
    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "3",
                "attribute_label": "Attr3",
            }
        ),
    ]

    config = DataExtractionConfig(selected_attribute_ids=["1", "3"])
    module = DataExtractionModule(config)

    filtered = module._filter_attributes(attributes)
    assert len(filtered) == 2
    assert filtered[0].attribute_id == "1"
    assert filtered[1].attribute_id == "3"


def test_prepare_context_full_document() -> None:
    """Test context preparation for full document."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "This is the full document content",
            "document_id": "123",
            "abstract": "This is the abstract",
        }
    )

    config = DataExtractionConfig(context_type=ContextType.FULL_DOCUMENT)
    module = DataExtractionModule(config)

    context = module._prepare_context(document)
    assert context == "This is the full document content"


def test_prepare_context_abstract_only() -> None:
    """Test context preparation for abstract only."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "This is the full document content",
            "document_id": "123",
            "abstract": "This is the abstract",
        }
    )

    config = DataExtractionConfig(context_type=ContextType.ABSTRACT_ONLY)
    module = DataExtractionModule(config)

    context = module._prepare_context(document)
    assert context == "This is the abstract"


def test_prepare_context_truncation() -> None:
    """Test context preparation with truncation."""
    long_content = "A" * 5000
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": long_content,
            "document_id": "123",
        }
    )

    config = DataExtractionConfig(max_context_length=1000)
    module = DataExtractionModule(config)

    context = module._prepare_context(document)
    assert len(context) <= 1003  # 1000 + "..."
    assert context.endswith("...")


def test_generate_prompt_single_attribute() -> None:
    """Test JSON input generation for single attribute."""
    attribute = EppiAttribute.model_validate(
        {
            "question_target": "",
            "output_data_type": "bool",
            "attribute_id": "1",
            "attribute_label": "Test Attribute",
            "attribute_set_description": "Test description",
        }
    )

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    payload_str = module._generate_user_message_json("Test context", [attribute])
    payload = json.loads(payload_str)

    assert "context" in payload and payload["context"] == "Test context"
    assert "attributes" in payload and isinstance(payload["attributes"], list)
    assert len(payload["attributes"]) == 1
    a0 = payload["attributes"][0]
    assert a0["attribute_id"] == "1"
    assert a0["attribute_label"] == "Test Attribute"
    assert a0["output_data_type"] == "bool"
    assert a0["question_target"] == "Test description"
    assert a0["attribute_set_description"] == "Test description"


def test_generate_prompt_multiple_attributes() -> None:
    """Test JSON input generation for multiple attributes."""
    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
                "attribute_set_description": "Desc1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
                "attribute_set_description": "Desc2",
            }
        ),
    ]

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    payload_str = module._generate_prompt("Test context", attributes)
    payload = json.loads(payload_str)
    assert payload["context"] == "Test context"
    assert len(payload["attributes"]) == 2
    ids = [a["attribute_id"] for a in payload["attributes"]]
    labels = [a["attribute_label"] for a in payload["attributes"]]
    descs = [a["attribute_set_description"] for a in payload["attributes"]]
    assert ids == ["1", "2"]
    assert labels == ["Attr1", "Attr2"]
    assert descs == ["Desc1", "Desc2"]


def test_parse_llm_response() -> None:
    """Test parsing LLM response."""
    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
    ]

    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )
    llm_response = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": "1",
                    "output_data": True,
                    "additional_text": "Found in document",
                    "reasoning": "Clear evidence",
                },
                {
                    "attribute_id": "2",
                    "output_data": False,
                    "additional_text": None,
                    "reasoning": "Not found",
                },
            ]
        }
    )

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    annotations = module._parse_llm_response(llm_response, attributes, document)

    assert len(annotations) == 2
    assert annotations[0].attribute.attribute_id == "1"
    assert annotations[0].output_data is True
    assert annotations[0].annotation_type == AnnotationType.LLM
    assert annotations[0].additional_text == "Found in document"
    assert annotations[0].reasoning == "Clear evidence"

    assert annotations[1].attribute.attribute_id == "2"
    assert annotations[1].output_data is False
    assert annotations[1].additional_text is None
    assert annotations[1].reasoning == "Not found"


def test_parse_llm_response_invalid_json() -> None:
    """Test parsing invalid JSON response raises ValueError."""
    attributes: list[EppiAttribute] = []
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    with pytest.raises(ValueError, match="Invalid JSON"):
        module._parse_llm_response("invalid json", attributes, document)


def test_parse_llm_response_invalid_schema() -> None:
    """Test parsing LLM response with invalid schema raises ValidationError."""
    from pydantic import ValidationError

    attributes: list[EppiAttribute] = []
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    # Invalid response - missing required fields
    invalid_response = json.dumps({"invalid": "structure"})

    config = DataExtractionConfig()
    module = DataExtractionModule(config)

    with pytest.raises(ValidationError):
        module._parse_llm_response(invalid_response, attributes, document)


def test_extract_from_document_with_mock() -> None:
    """Test extract_from_document with mocked LLM call."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
    ]

    mock_response = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": "1",
                    "output_data": True,
                    "additional_text": "Found",
                    "reasoning": "Evidence",
                },
            ]
        }
    )

    with patch(
        "app.extractors.data_extraction_module.litellm.completion"
    ) as mock_completion:
        mock_choice = type(
            "MockChoice",
            (),
            {"message": type("MockMessage", (), {"content": mock_response})()},
        )()
        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        config = DataExtractionConfig()
        module = DataExtractionModule(config)

        annotations = module.extract_from_document(document, attributes)

        assert len(annotations) == 1
        assert annotations[0].output_data is True
        assert annotations[0].annotation_type == AnnotationType.LLM


def test_convenience_function_extract_single_attribute() -> None:
    """Test convenience function for single attribute extraction."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    attribute = EppiAttribute.model_validate(
        {
            "question_target": "",
            "output_data_type": "bool",
            "attribute_id": "1",
            "attribute_label": "Attr1",
        }
    )

    mock_response = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": "1",
                    "output_data": True,
                    "additional_text": "Found",
                    "reasoning": "Evidence",
                },
            ]
        }
    )

    with patch(
        "app.extractors.data_extraction_module.litellm.completion"
    ) as mock_completion:
        mock_choice = type(
            "MockChoice",
            (),
            {"message": type("MockMessage", (), {"content": mock_response})()},
        )()
        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        result = extract_single_attribute(document, attribute)

        assert result is not None
        assert result.output_data is True
        assert result.annotation_type == AnnotationType.LLM


def test_convenience_function_extract_all_attributes() -> None:
    """Test convenience function for all attributes extraction."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
    ]

    mock_response = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": "1",
                    "output_data": True,
                    "additional_text": "Found",
                    "reasoning": "Evidence",
                },
                {
                    "attribute_id": "2",
                    "output_data": False,
                    "additional_text": None,
                    "reasoning": "Not found",
                },
            ]
        }
    )

    with patch(
        "app.extractors.data_extraction_module.litellm.completion"
    ) as mock_completion:
        mock_choice = type(
            "MockChoice",
            (),
            {"message": type("MockMessage", (), {"content": mock_response})()},
        )()
        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        results = extract_all_attributes(document, attributes)

        assert len(results) == 2
        assert results[0].output_data is True
        assert results[1].output_data is False


def test_convenience_function_extract_batch_attributes() -> None:
    """Test convenience function for batch attributes extraction."""
    document = EppiDocument.model_validate(
        {
            "name": "Test Document",
            "citation": {"id": str(uuid4()), "visibility": "public"},
            "context": "Test context",
            "document_id": "123",
        }
    )

    attributes = [
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "1",
                "attribute_label": "Attr1",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "2",
                "attribute_label": "Attr2",
            }
        ),
        EppiAttribute.model_validate(
            {
                "question_target": "",
                "output_data_type": "bool",
                "attribute_id": "3",
                "attribute_label": "Attr3",
            }
        ),
    ]

    mock_response = json.dumps(
        {
            "annotations": [
                {
                    "attribute_id": "1",
                    "output_data": True,
                    "additional_text": "Found",
                    "reasoning": "Evidence",
                },
                {
                    "attribute_id": "3",
                    "output_data": False,
                    "additional_text": None,
                    "reasoning": "Not found",
                },
            ]
        }
    )

    with patch(
        "app.extractors.data_extraction_module.litellm.completion"
    ) as mock_completion:
        mock_choice = type(
            "MockChoice",
            (),
            {"message": type("MockMessage", (), {"content": mock_response})()},
        )()

        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        results = extract_batch_attributes(document, attributes, ["1", "3"])

        assert len(results) == 2
        assert results[0].attribute.attribute_id == "1"
        assert results[1].attribute.attribute_id == "3"
