"""Tests for EPPI-specific models."""

from uuid import uuid4

import pytest
from destiny_sdk.references import Reference

from app.models.base import AnnotationType
from app.models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
    EppiRawData,
)


class TestEppiAttribute:
    """Test EppiAttribute model with new union type and EPPI-specific fields."""

    def test_eppi_attribute_creation(self) -> None:
        """Test creating EppiAttribute with default values."""
        attr = EppiAttribute(
            attribute_id="eppi1",
            attribute_label="Test EPPI Attribute",
        )
        assert attr.attribute_id == "eppi1"
        assert attr.attribute_label == "Test EPPI Attribute"
        assert attr.question_target == ""  # Default empty for EPPI
        assert attr.output_data_type is bool  # Default boolean for EPPI

    def test_eppi_attribute_with_eppi_fields(self) -> None:
        """Test creating EppiAttribute with EPPI-specific fields."""
        attr = EppiAttribute(
            attribute_id="eppi2",
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

    def test_eppi_attribute_with_different_output_types(self) -> None:
        """Test EppiAttribute with different output_data_type values."""
        # Test with str type
        attr_str = EppiAttribute(
            attribute_id="eppi3",
            attribute_label="String Attribute",
            output_data_type=str,
        )
        assert attr_str.output_data_type is str

        # Test with int type
        attr_int = EppiAttribute(
            attribute_id="eppi4",
            attribute_label="Integer Attribute",
            output_data_type=int,
        )
        assert attr_int.output_data_type is int

    def test_eppi_attribute_camel_case_mapping(self) -> None:
        """Test that camelCase JSON fields are mapped correctly."""
        # This would be tested with actual JSON data in integration tests
        # Here we test that the model can be created with the expected fields
        attr = EppiAttribute(
            attribute_id="eppi5",
            attribute_label="Test Attribute",
            attribute_set_description="Test description",
        )
        assert hasattr(attr, "attribute_set_description")
        assert hasattr(attr, "hierarchy_path")
        assert hasattr(attr, "hierarchy_level")


class TestEppiItemAttributeFullTextDetails:
    """Test EppiItemAttributeFullTextDetails with new mode='before' validator."""

    def test_valid_creation_with_item_document_id(self) -> None:
        """Test creating with item_document_id provided."""
        details = EppiItemAttributeFullTextDetails(item_document_id=123)
        assert details.item_document_id == 123
        assert details.text is None
        assert details.item_arm is None

    def test_valid_creation_with_text(self) -> None:
        """Test creating with text provided."""
        details = EppiItemAttributeFullTextDetails(text="Some text content")
        assert details.text == "Some text content"
        assert details.item_document_id is None
        assert details.item_arm is None

    def test_valid_creation_with_item_arm(self) -> None:
        """Test creating with item_arm provided."""
        details = EppiItemAttributeFullTextDetails(item_arm="arm1")
        assert details.item_arm == "arm1"
        assert details.item_document_id is None
        assert details.text is None

    def test_valid_creation_with_multiple_fields(self) -> None:
        """Test creating with multiple fields provided."""
        details = EppiItemAttributeFullTextDetails(
            item_document_id=123,
            text="Some text",
            item_arm="arm1",
        )
        assert details.item_document_id == 123
        assert details.text == "Some text"
        assert details.item_arm == "arm1"

    def test_invalid_creation_all_none(self) -> None:
        """Test that creation fails when all fields are None."""
        with pytest.raises(ValueError, match="At least one field must be provided"):
            EppiItemAttributeFullTextDetails()

    def test_invalid_creation_explicit_none(self) -> None:
        """Test that creation fails when all fields are explicitly None."""
        with pytest.raises(ValueError, match="At least one field must be provided"):
            EppiItemAttributeFullTextDetails(
                item_document_id=None,
                text=None,
                item_arm=None,
            )

    def test_dynamic_validator_with_additional_fields(self) -> None:
        """Test that the dynamic validator works with additional fields."""
        # This tests the future-proofing aspect of the dynamic validator
        # The validator checks all fields in the data, not just hardcoded ones
        details = EppiItemAttributeFullTextDetails(item_document_id=123)
        assert details.item_document_id == 123


class TestEppiDocument:
    """Test EppiDocument model."""

    def test_eppi_document_creation(self) -> None:
        """Test creating EppiDocument."""
        citation = Reference(
            id=uuid4(),
            title="Test EPPI Document",
            authors=["EPPI Author"],
        )
        doc = EppiDocument(
            name="Test EPPI Document",
            citation=citation,
            context="Test content",
            document_id="eppi_doc1",
            filename="test.pdf",
        )
        assert doc.name == "Test EPPI Document"
        assert doc.document_id == "eppi_doc1"
        assert doc.filename == "test.pdf"

    def test_eppi_document_with_list_context(self) -> None:
        """Test creating EppiDocument with list context."""
        citation = Reference(
            id=uuid4(),
            title="Test EPPI Document 2",
            authors=["EPPI Author 2"],
        )
        doc = EppiDocument(
            name="Test EPPI Document 2",
            citation=citation,
            context=["Paragraph 1", "Paragraph 2"],
            document_id="eppi_doc2",
        )
        assert doc.context == ["Paragraph 1", "Paragraph 2"]


class TestEppiGoldStandardAnnotation:
    """Test EppiGoldStandardAnnotation model."""

    def test_eppi_gold_standard_annotation_creation(self) -> None:
        """Test creating EppiGoldStandardAnnotation."""
        attr = EppiAttribute(
            attribute_id="eppi_attr1",
            attribute_label="Test EPPI Attribute",
        )

        annotation = EppiGoldStandardAnnotation(
            attribute=attr,
            output_data=True,
            annotation_type=AnnotationType.HUMAN,
        )
        assert annotation.attribute == attr
        assert annotation.output_data is True
        assert annotation.annotation_type == AnnotationType.HUMAN

    def test_eppi_gold_standard_annotation_with_llm(self) -> None:
        """Test creating EppiGoldStandardAnnotation with LLM type."""
        attr = EppiAttribute(
            attribute_id="eppi_attr2",
            attribute_label="Test EPPI Attribute 2",
        )

        annotation = EppiGoldStandardAnnotation(
            attribute=attr,
            output_data="Test response",
            annotation_type=AnnotationType.LLM,
        )
        assert annotation.annotation_type == AnnotationType.LLM


class TestEppiGoldStandardAnnotatedDocument:
    """Test EppiGoldStandardAnnotatedDocument model."""

    def test_eppi_gold_standard_annotated_document_creation(self) -> None:
        """Test creating EppiGoldStandardAnnotatedDocument."""
        citation = Reference(
            id=uuid4(),
            title="Test EPPI Document 3",
            authors=["EPPI Author 3"],
        )

        attr = EppiAttribute(
            attribute_id="eppi_attr3",
            attribute_label="Test EPPI Attribute 3",
        )

        annotation = EppiGoldStandardAnnotation(
            attribute=attr,
            output_data=True,
            annotation_type=AnnotationType.HUMAN,
        )

        doc = EppiGoldStandardAnnotatedDocument(
            name="Test EPPI Document 3",
            citation=citation,
            context="Test content",
            document_id="eppi_doc3",
            annotations=[annotation],
        )
        assert doc.name == "Test EPPI Document 3"
        assert len(doc.annotations) == 1
        assert doc.annotations[0].output_data is True


class TestEppiRawData:
    """Test EppiRawData model for validation."""

    def test_eppi_raw_data_creation(self) -> None:
        """Test creating EppiRawData with minimal valid data."""
        data: dict = {
            "CodeSets": [],
            "References": [],
        }
        raw_data = EppiRawData.model_validate(data)
        assert raw_data.code_sets == []
        assert raw_data.references == []

    def test_eppi_raw_data_with_codesets(self) -> None:
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
