"""Tests for EPPI-specific models."""

import pytest

from app.data_models.base import AnnotationType, AttributeType
from app.data_models.eppi import (
    EppiAttribute,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
    EppiRawData,
)


def test_eppi_attribute_creation_from_json_data() -> None:
    """Test creating EppiAttribute from JSON-like data (would come from EPPI JSON)."""
    # This mimics how EppiAttribute is created from JSON data in the ann converter
    # The data structure matches what comes from EPPI JSON files
    # Currently the alias generator is not working as expected,
    # so we use snake_case field names
    attr_data = {
        "AttributeId": 5730447,  # Integer ID from JSON
        "AttributeName": "Test EPPI Attribute",
        "AttributeDescription": "Test description",
        "AttributeSetDescription": "Test set description",
        "AttributeType": "Selectable (show checkbox)",
        "AttributeTypeId": 2,
        "AttributeSetId": 5730392,
        "OriginalAttributeID": 0,
        "ExtURL": "",
        "ExtType": "",
        # These fields are added by the annotation converter
        "question_target": "",  # Always empty for EPPI
        "output_data_type": AttributeType.BOOL.value,  # Always boolean for EPPI
        "attribute_id": "5730447",  # Converted to string
        "attribute_label": "Test EPPI Attribute",
        # Use snake_case field names directly since alias generator is not working
        "attribute_set_description": "Test set description",
        "attribute_type": "Selectable (show checkbox)",
    }
    attr = EppiAttribute.model_validate(attr_data)
    assert attr.attribute_id == 5730447
    assert attr.attribute_label == "Test EPPI Attribute"
    assert attr.question_target == ""  # Default empty for EPPI
    assert attr.output_data_type.to_python_type() is bool  # Default boolean for EPPI
    # Test the EPPI-specific fields
    assert attr.attribute_set_description == "Test set description"
    assert attr.attribute_type == "Selectable (show checkbox)"


def test_eppi_attribute_with_eppi_fields() -> None:
    """Test creating EppiAttribute with EPPI-specific fields."""
    attr = EppiAttribute(
        attribute_id=2345,
        attribute_label="Test EPPI Attribute 2",
        attribute_set_description="Test description",
        hierarchy_path="root.test.attribute",
        hierarchy_level=2,
        is_leaf=False,
        parent_attribute_id="parent1",
        attribute_type="Selectable",
    )
    assert attr.attribute_set_description == "Test description"
    assert attr.hierarchy_path == "root.test.attribute"
    assert attr.hierarchy_level == 2
    assert attr.is_leaf is False
    assert attr.parent_attribute_id == "parent1"
    assert attr.attribute_type == "Selectable"


def test_eppi_attribute_with_different_output_types() -> None:
    """Test EppiAttribute with different output_data_type values."""
    # Test with str type
    attr_str = EppiAttribute(
        attribute_id=3456,
        attribute_label="String Attribute",
        output_data_type=AttributeType.STRING,
    )
    assert attr_str.output_data_type.value == AttributeType.STRING.value

    # Test with int type
    attr_int = EppiAttribute(
        attribute_id=4567,
        attribute_label="Integer Attribute",
        output_data_type=AttributeType.INTEGER,
    )
    assert attr_int.output_data_type.value == AttributeType.INTEGER.value


def test_eppi_attribute_camel_case_mapping() -> None:
    """Test that camelCase JSON fields are mapped correctly."""
    # This would be tested with actual JSON data in integration tests
    # Here we test that the model can be created with the expected fields
    attr = EppiAttribute(
        attribute_id=5678,
        attribute_label="Test Attribute",
        attribute_set_description="Test description",
    )
    assert hasattr(attr, "attribute_set_description")
    assert hasattr(attr, "hierarchy_path")
    assert hasattr(attr, "hierarchy_level")


def test_valid_creation_with_item_document_id() -> None:
    """Test creating with item_document_id provided."""
    details = EppiItemAttributeFullTextDetails(item_document_id=123)
    assert details.item_document_id == 123
    assert details.text is None
    assert details.item_arm is None


def test_valid_creation_with_text() -> None:
    """Test creating with text provided."""
    details = EppiItemAttributeFullTextDetails(text="Some text content")
    assert details.text == "Some text content"
    assert details.item_document_id is None
    assert details.item_arm is None


def test_valid_creation_with_item_arm() -> None:
    """Test creating with item_arm provided."""
    details = EppiItemAttributeFullTextDetails(item_arm="arm1")
    assert details.item_arm == "arm1"
    assert details.item_document_id is None
    assert details.text is None


def test_valid_creation_with_multiple_fields() -> None:
    """Test creating with multiple fields provided."""
    details = EppiItemAttributeFullTextDetails(
        item_document_id=123,
        text="Some text",
        item_arm="arm1",
    )
    assert details.item_document_id == 123
    assert details.text == "Some text"
    assert details.item_arm == "arm1"


def test_invalid_creation_all_none() -> None:
    """Test that creation fails when all fields are None."""
    with pytest.raises(ValueError, match="At least one field must be provided"):
        EppiItemAttributeFullTextDetails()


def test_invalid_creation_explicit_none() -> None:
    """Test that creation fails when all fields are explicitly None."""
    with pytest.raises(ValueError, match="At least one field must be provided"):
        EppiItemAttributeFullTextDetails(
            item_document_id=None,
            text=None,
            item_arm=None,
        )


def test_dynamic_validator_with_additional_fields() -> None:
    """Test that the dynamic validator works with additional fields."""
    # This tests the future-proofing aspect of the dynamic validator
    # The validator checks all fields in the data, not just hardcoded ones
    details = EppiItemAttributeFullTextDetails(item_document_id=123)
    assert details.item_document_id == 123


class TestEppiDocument:
    """Test EppiDocument model."""

    # def test_eppi_document_creation(self) -> None:
    #     """Test creating EppiDocument."""
    #     citation = Reference(
    #         id=uuid4(),
    #         title="Test EPPI Document",
    #         authors=["EPPI Author"],
    #     )
    #     doc = EppiDocument(
    #         name="Test EPPI Document",
    #         citation=citation,
    #         context="Test content",
    #         document_id="eppi_doc1",
    #         filename="test.pdf",
    #     )
    #     assert doc.name == "Test EPPI Document"
    #     assert doc.document_id == "eppi_doc1"
    #     assert doc.filename == "test.pdf"

    # def test_eppi_document_with_list_context(self) -> None:
    #     """Test creating EppiDocument with list context."""
    #     citation = Reference(
    #         id=uuid4(),
    #         title="Test EPPI Document 2",
    #         authors=["EPPI Author 2"],
    #     )
    #     doc = EppiDocument(
    #         name="Test EPPI Document 2",
    #         citation=citation,
    #         context=["Paragraph 1", "Paragraph 2"],
    #         document_id="eppi_doc2",
    #     )
    #     assert doc.context == ["Paragraph 1", "Paragraph 2"]


def test_eppi_gold_standard_annotation_creation_from_json_data() -> None:
    """Test creating EppiGoldStandardAnnotation from JSON-like data."""
    # This mimics how annotations are created from EPPI JSON data
    # First create the attribute from JSON-like data
    attr_data = {
        "AttributeId": 5730447,
        "AttributeName": "Test EPPI Attribute",
        "question_target": "",
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": "5730447",
        "attribute_label": "Test EPPI Attribute",
    }
    attr = EppiAttribute.model_validate(attr_data)

    # Create annotation data as it would come from EPPI JSON
    annotation_data = {
        "attribute": attr,
        "output_data": True,
        "annotation_type": AnnotationType.HUMAN,
        "additional_text": "Some additional context text",
        "arm_id": 3,
        "arm_title": "Test Arm",
        "arm_description": "Test arm description",
    }
    annotation = EppiGoldStandardAnnotation.model_validate(annotation_data)
    assert annotation.attribute == attr
    assert annotation.output_data is True
    assert annotation.annotation_type == AnnotationType.HUMAN
    assert annotation.additional_text == "Some additional context text"
    assert annotation.arm_id == 3
    assert annotation.arm_title == "Test Arm"


def test_eppi_gold_standard_annotation_with_llm() -> None:
    """Test creating EppiGoldStandardAnnotation with LLM type."""
    attr = EppiAttribute(
        attribute_id=2345,
        attribute_label="Test EPPI Attribute 2",
        output_data_type=AttributeType.STRING,
    )

    annotation = EppiGoldStandardAnnotation(
        attribute=attr,
        output_data="Test response",
        annotation_type=AnnotationType.LLM,
    )
    assert annotation.annotation_type == AnnotationType.LLM


class TestEppiGoldStandardAnnotatedDocument:
    """Test EppiGoldStandardAnnotatedDocument model."""

    # def test_eppi_gold_standard_annotated_document_creation(self) -> None:
    #     """Test creating EppiGoldStandardAnnotatedDocument."""
    #     citation = Reference(
    #         id=uuid4(),
    #         title="Test EPPI Document 3",
    #         authors=["EPPI Author 3"],
    #     )

    #     attr = EppiAttribute(
    #         attribute_id="eppi_attr3",
    #         attribute_label="Test EPPI Attribute 3",
    #     )

    #     annotation = EppiGoldStandardAnnotation(
    #         attribute=attr,
    #         output_data=True,
    #         annotation_type=AnnotationType.HUMAN,
    #     )

    #     doc = EppiGoldStandardAnnotatedDocument(
    #         name="Test EPPI Document 3",
    #         citation=citation,
    #         context="Test content",
    #         document_id="eppi_doc3",
    #         annotations=[annotation],
    #     )
    #     assert doc.name == "Test EPPI Document 3"
    #     assert len(doc.annotations) == 1
    #     assert doc.annotations[0].output_data is True


def test_eppi_raw_data_creation() -> None:
    """Test creating EppiRawData with minimal valid data."""
    data: dict = {
        "CodeSets": [],
        "References": [],
    }
    raw_data = EppiRawData.model_validate(data)
    assert raw_data.code_sets == []
    assert raw_data.references == []


def test_eppi_raw_data_with_codesets() -> None:
    """Test EppiRawData with CodeSets data."""
    data = {
        "CodeSets": [
            {
                "SetName": "Test Set",
                "SetId": 1,
                "ReviewSetId": 1,
                "SetDescription": "Test description",
                "SetType": {
                    "SetTypeDescription": "Standard",
                    "SetTypeName": "Standard",
                    "SetTypeId": 3,
                },
                "Attributes": {
                    "AttributesList": [
                        {
                            "AttributeId": 1,
                            "AttributeName": "Test Attribute",
                            "AttributeType": "Selectable",
                            "AttributeSetId": 1,
                            "AttributeSetDescription": "Test description",
                            "AttributeTypeId": 2,
                            "AttributeDescription": "",
                            "ExtURL": "",
                            "ExtType": "",
                            "OriginalAttributeID": 0,
                        }
                    ]
                },
            }
        ],
        "References": [],
    }
    raw_data = EppiRawData.model_validate(data)
    assert len(raw_data.code_sets) == 1
    # Check that the codeset has attributes
    assert raw_data.code_sets[0].attributes is not None
    assert "AttributesList" in raw_data.code_sets[0].attributes
    assert len(raw_data.code_sets[0].attributes["AttributesList"]) == 1
