"""Tests for the new reasoning field in EppiGoldStandardAnnotation."""

from app.models.base import AnnotationType
from app.models.eppi import EppiAttribute, EppiGoldStandardAnnotation


def test_eppi_gold_standard_annotation_with_reasoning() -> None:
    """Test creating EppiGoldStandardAnnotation with reasoning field."""
    # Create a test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": "bool",
        "attribute_id": "5730447",
        "attribute_label": "Test Attribute",
        "attribute_set_description": "Test set description",
        "attribute_type": "Selectable (show checkbox)",
    }
    attribute = EppiAttribute.model_validate(attr_data)

    # Create annotation with reasoning
    annotation_data = {
        "attribute": attribute,
        "output_data": True,
        "annotation_type": AnnotationType.LLM,
        "additional_text": "Supporting citation text",
        "reasoning": "The document clearly mentions this attribute in the methodology section",
        "arm_id": 1,
        "arm_title": "Test Arm",
    }

    annotation = EppiGoldStandardAnnotation.model_validate(annotation_data)

    # Verify all fields including the new reasoning field
    assert annotation.attribute.attribute_id == "5730447"
    assert annotation.output_data is True
    assert annotation.annotation_type == AnnotationType.LLM
    assert annotation.additional_text == "Supporting citation text"
    assert (
        annotation.reasoning
        == "The document clearly mentions this attribute in the methodology section"
    )
    assert annotation.arm_id == 1
    assert annotation.arm_title == "Test Arm"


def test_eppi_gold_standard_annotation_without_reasoning() -> None:
    """Test creating EppiGoldStandardAnnotation without reasoning field (backward compatibility)."""
    # Create a test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": "bool",
        "attribute_id": "5730448",
        "attribute_label": "Test Attribute 2",
        "attribute_set_description": "Test set description 2",
        "attribute_type": "Selectable (show checkbox)",
    }
    attribute = EppiAttribute.model_validate(attr_data)

    # Create annotation without reasoning (should default to None)
    annotation_data = {
        "attribute": attribute,
        "output_data": False,
        "annotation_type": AnnotationType.HUMAN,
        "additional_text": "Human annotation citation",
        "arm_id": 2,
        "arm_title": "Control Arm",
    }

    annotation = EppiGoldStandardAnnotation.model_validate(annotation_data)

    # Verify all fields including reasoning defaults to None
    assert annotation.attribute.attribute_id == "5730448"
    assert annotation.output_data is False
    assert annotation.annotation_type == AnnotationType.HUMAN
    assert annotation.additional_text == "Human annotation citation"
    assert annotation.reasoning is None  # Should default to None
    assert annotation.arm_id == 2
    assert annotation.arm_title == "Control Arm"


def test_eppi_gold_standard_annotation_reasoning_optional() -> None:
    """Test that reasoning field is optional and can be explicitly set to None."""
    # Create a test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": "bool",
        "attribute_id": "5730449",
        "attribute_label": "Test Attribute 3",
        "attribute_set_description": "Test set description 3",
        "attribute_type": "Selectable (show checkbox)",
    }
    attribute = EppiAttribute.model_validate(attr_data)

    # Create annotation with explicit None reasoning
    annotation_data = {
        "attribute": attribute,
        "output_data": True,
        "annotation_type": AnnotationType.LLM,
        "additional_text": "Another citation",
        "reasoning": None,  # Explicitly set to None
        "arm_id": 3,
        "arm_title": "Treatment Arm",
    }

    annotation = EppiGoldStandardAnnotation.model_validate(annotation_data)

    # Verify reasoning is None
    assert annotation.reasoning is None
    assert annotation.output_data is True
    assert annotation.annotation_type == AnnotationType.LLM


def test_eppi_gold_standard_annotation_serialization() -> None:
    """Test that the reasoning field serializes correctly to JSON."""
    # Create a test attribute
    attr_data = {
        "question_target": "",
        "output_data_type": "bool",
        "attribute_id": "5730450",
        "attribute_label": "Test Attribute 4",
        "attribute_set_description": "Test set description 4",
        "attribute_type": "Selectable (show checkbox)",
    }
    attribute = EppiAttribute.model_validate(attr_data)

    # Create annotation with reasoning
    annotation_data = {
        "attribute": attribute,
        "output_data": False,
        "annotation_type": AnnotationType.LLM,
        "additional_text": "Serialization test citation",
        "reasoning": "This is a test reasoning for serialization",
        "arm_id": 4,
        "arm_title": "Placebo Arm",
    }

    annotation = EppiGoldStandardAnnotation.model_validate(annotation_data)

    # Test serialization
    json_data = annotation.model_dump()

    # Verify reasoning field is in the serialized data
    assert "reasoning" in json_data
    assert json_data["reasoning"] == "This is a test reasoning for serialization"
    assert json_data["output_data"] is False
    assert json_data["annotation_type"] == "llm"
    assert json_data["additional_text"] == "Serialization test citation"
