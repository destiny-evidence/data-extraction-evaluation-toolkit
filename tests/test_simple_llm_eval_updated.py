"""Tests for the updated simple_llm_eval.py script."""

import json
from unittest.mock import patch

from app.models.base import AnnotationType
from app.models.eppi import EppiAttribute, EppiGoldStandardAnnotation
from app.scripts.simple_llm_eval import call_llm


def test_call_llm_with_mock_response() -> None:
    """Test call_llm function with mocked LLM response."""
    # Create test attributes
    attr1_data = {
        "question_target": "",
        "output_data_type": bool,
        "attribute_id": "5730447",
        "attribute_label": "Test Attribute 1",
        "attribute_set_description": "Test description 1",
        "attribute_type": "Selectable (show checkbox)",
    }
    attr2_data = {
        "question_target": "",
        "output_data_type": bool,
        "attribute_id": "5730448",
        "attribute_label": "Test Attribute 2",
        "attribute_set_description": "Test description 2",
        "attribute_type": "Selectable (show checkbox)",
    }

    attribute1 = EppiAttribute.model_validate(attr1_data)
    attribute2 = EppiAttribute.model_validate(attr2_data)
    attributes = [attribute1, attribute2]

    # Mock LLM response
    mock_response = {
        "annotations": [
            {
                "attribute_id": "5730447",
                "output_data": True,
                "annotation_type": "llm",
                "additional_text": "Found in methodology section",
                "reasoning": "The document clearly describes this attribute",
            },
            {
                "attribute_id": "5730448",
                "output_data": False,
                "annotation_type": "llm",
                "additional_text": None,
                "reasoning": "No mention of this attribute found",
            },
        ]
    }

    # Mock the litellm.completion call
    with patch("app.scripts.simple_llm_eval.litellm.completion") as mock_completion:
        # Create mock response object
        mock_choice = type(
            "MockChoice",
            (),
            {
                "message": type(
                    "MockMessage", (), {"content": json.dumps(mock_response)}
                )()
            },
        )()

        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        # Call the function
        result = call_llm("Test document context", attributes)

        # Verify the result
        assert len(result) == 2
        assert all(isinstance(ann, EppiGoldStandardAnnotation) for ann in result)

        # Check first annotation
        ann1 = result[0]
        assert ann1.attribute.attribute_id == "5730447"
        assert ann1.output_data is True
        assert ann1.annotation_type == AnnotationType.LLM
        assert ann1.additional_text == "Found in methodology section"
        assert ann1.reasoning == "The document clearly describes this attribute"

        # Check second annotation
        ann2 = result[1]
        assert ann2.attribute.attribute_id == "5730448"
        assert ann2.output_data is False
        assert ann2.annotation_type == AnnotationType.LLM
        assert ann2.additional_text is None
        assert ann2.reasoning == "No mention of this attribute found"


def test_call_llm_with_missing_attribute() -> None:
    """Test call_llm function when LLM returns attribute_id not in input attributes."""
    # Create test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": bool,
        "attribute_id": "5730447",
        "attribute_label": "Test Attribute",
        "attribute_set_description": "Test description",
        "attribute_type": "Selectable (show checkbox)",
    }

    attribute = EppiAttribute.model_validate(attr_data)
    attributes = [attribute]

    # Mock LLM response with unknown attribute_id
    mock_response = {
        "annotations": [
            {
                "attribute_id": "5730447",  # This one exists
                "output_data": True,
                "annotation_type": "llm",
                "additional_text": "Found in document",
                "reasoning": "Clear evidence found",
            },
            {
                "attribute_id": "9999999",  # This one doesn't exist
                "output_data": False,
                "annotation_type": "llm",
                "additional_text": None,
                "reasoning": "Not found",
            },
        ]
    }

    # Mock the litellm.completion call
    with patch("app.scripts.simple_llm_eval.litellm.completion") as mock_completion:
        # Create mock response object
        mock_choice = type(
            "MockChoice",
            (),
            {
                "message": type(
                    "MockMessage", (), {"content": json.dumps(mock_response)}
                )()
            },
        )()

        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        # Call the function
        result = call_llm("Test document context", attributes)

        # Verify only the valid attribute is included
        assert len(result) == 1
        assert result[0].attribute.attribute_id == "5730447"
        assert result[0].output_data is True


def test_call_llm_with_empty_response() -> None:
    """Test call_llm function with empty LLM response."""
    # Create test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": bool,
        "attribute_id": "5730447",
        "attribute_label": "Test Attribute",
        "attribute_set_description": "Test description",
        "attribute_type": "Selectable (show checkbox)",
    }

    attribute = EppiAttribute.model_validate(attr_data)
    attributes = [attribute]

    # Mock LLM response with empty annotations
    mock_response: dict[str, list] = {"annotations": []}

    # Mock the litellm.completion call
    with patch("app.scripts.simple_llm_eval.litellm.completion") as mock_completion:
        # Create mock response object
        mock_choice = type(
            "MockChoice",
            (),
            {
                "message": type(
                    "MockMessage", (), {"content": json.dumps(mock_response)}
                )()
            },
        )()

        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        # Call the function
        result = call_llm("Test document context", attributes)

        # Verify empty result
        assert len(result) == 0


def test_call_llm_prompt_generation() -> None:
    """Test that the prompt is generated correctly with attribute information."""
    # Create test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": bool,
        "attribute_id": "5730447",
        "attribute_label": "Test Attribute",
        "attribute_set_description": "Test description",
        "attribute_type": "Selectable (show checkbox)",
    }

    attribute = EppiAttribute.model_validate(attr_data)
    attributes = [attribute]

    # Mock the litellm.completion call
    with patch("app.scripts.simple_llm_eval.litellm.completion") as mock_completion:
        # Create mock response object
        mock_choice = type(
            "MockChoice",
            (),
            {
                "message": type(
                    "MockMessage", (), {"content": json.dumps({"annotations": []})}
                )()
            },
        )()

        mock_completion.return_value = type(
            "MockResponse", (), {"choices": [mock_choice]}
        )()

        # Call the function
        call_llm("Test document context", attributes)

        # Verify the call was made with correct parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args

        # Check that the prompt contains attribute information
        messages = call_args[1]["messages"]
        user_message = messages[1]["content"]

        assert "Test Attribute" in user_message
        assert "5730447" in user_message
        assert "Test description" in user_message
        assert "output_data" in user_message
        assert "annotation_type" in user_message
        assert "reasoning" in user_message
