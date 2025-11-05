"""Tests for core base models."""

from uuid import uuid4

from destiny_sdk.references import Reference

from app.models.base import (
    AnnotationType,
    Attribute,
    AttributesList,
    Document,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotation,
)


def test_attribute_creation_from_dict() -> None:
    """Test creating attribute from dictionary data (as would come from JSON)."""
    # This mimics how attributes are created from JSON data in the annotation converter
    attr_data = {
        "question_target": "Is this a test?",
        "output_data_type": "bool",
        "attribute_id": "test1",
        "attribute_label": "Test Boolean Attribute",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.question_target == "Is this a test?"
    assert attr.output_data_type is bool
    assert attr.attribute_id == "test1"
    assert attr.attribute_label == "Test Boolean Attribute"


def test_attribute_creation_with_different_types() -> None:
    """Test creating attributes with different output_data_type values from dict data."""
    # Test with str type
    attr_data_str = {
        "question_target": "What is the name?",
        "output_data_type": "str",
        "attribute_id": "test2",
        "attribute_label": "Test String Attribute",
    }
    attr_str = Attribute.model_validate(attr_data_str)
    assert attr_str.output_data_type is str

    # Test with int type
    attr_data_int = {
        "question_target": "How many items?",
        "output_data_type": int,
        "attribute_id": "test3",
        "attribute_label": "Test Integer Attribute",
    }
    attr_int = Attribute.model_validate(attr_data_int)
    assert attr_int.output_data_type is int

    # Test with list type
    attr_data_list = {
        "question_target": "What are the items?",
        "output_data_type": list,
        "attribute_id": "test4",
        "attribute_label": "Test List Attribute",
    }
    attr_list = Attribute.model_validate(attr_data_list)
    assert attr_list.output_data_type is list

    # Test with dict type
    attr_data_dict = {
        "question_target": "What are the details?",
        "output_data_type": dict,
        "attribute_id": "test5",
        "attribute_label": "Test Dictionary Attribute",
    }
    attr_dict = Attribute.model_validate(attr_data_dict)
    assert attr_dict.output_data_type is dict

    # Test with float type
    attr_data_float = {
        "question_target": "What is the value?",
        "output_data_type": float,
        "attribute_id": "test6",
        "attribute_label": "Test Float Attribute",
    }
    attr_float = Attribute.model_validate(attr_data_float)
    assert attr_float.output_data_type is float


def test_attribute_validation_required_fields() -> None:
    """Test that required fields are validated when creating from dict data."""
    # Test that we can create attributes with valid data
    attr_data = {
        "question_target": "Test",
        "output_data_type": "bool",
        "attribute_id": "test_id",
        "attribute_label": "Test Label",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.question_target == "Test"
    assert attr.attribute_id == "test_id"
    assert attr.attribute_label == "Test Label"


def test_attributes_list_creation() -> None:
    """Test creating AttributesList with multiple attributes."""
    attrs = [
        Attribute(
            question_target="Question 1",
            output_data_type="bool",
            attribute_id="attr1",
            attribute_label="Attribute 1",
        ),
        Attribute(
            question_target="Question 2",
            output_data_type="str",
            attribute_id="attr2",
            attribute_label="Attribute 2",
        ),
    ]
    attr_list = AttributesList(attributes=attrs)
    assert len(attr_list.attributes) == 2
    assert attr_list.attributes[0].attribute_id == "attr1"
    assert attr_list.attributes[1].attribute_id == "attr2"


def test_attributes_list_iteration() -> None:
    """Test that AttributesList is iterable."""
    attrs = [
        Attribute(
            question_target="Question 1",
            output_data_type="bool",
            attribute_id="attr1",
            attribute_label="Attribute 1",
        ),
    ]
    attr_list = AttributesList(attributes=attrs)

    # Test iteration
    for attr in attr_list:
        assert attr.attribute_id == "attr1"

    # Test to_list method
    assert attr_list.to_list() == attrs


def test_document_creation_from_dict() -> None:
    """Test creating a document from dictionary data (as would come from JSON)."""
    # This mimics how documents are created from JSON data
    citation = Reference(
        id=uuid4(),
        title="Test Document",
        authors=["Test Author"],
    )

    doc_data = {
        "name": "Test Document",
        "citation": citation,
        "context": "This is test content",
        "document_id": "doc1",
        "filename": "test.pdf",
    }
    doc = Document.model_validate(doc_data)
    assert doc.name == "Test Document"
    assert doc.document_id == "doc1"
    assert doc.filename == "test.pdf"
    assert doc.context == "This is test content"


def test_document_creation_with_list_context_from_dict() -> None:
    """Test creating a document with list context from dictionary data."""
    citation = Reference(
        id=uuid4(),
        title="Test Document 2",
        authors=["Test Author 2"],
    )

    doc_data = {
        "name": "Test Document 2",
        "citation": citation,
        "context": ["Paragraph 1", "Paragraph 2"],
        "document_id": "doc2",
    }
    doc = Document.model_validate(doc_data)
    assert doc.context == ["Paragraph 1", "Paragraph 2"]


def test_gold_standard_annotation_creation_from_dict() -> None:
    """Test creating a gold standard annotation from dictionary data."""
    # This mimics how annotations are created from JSON data
    attr_data = {
        "question_target": "Test question",
        "output_data_type": "bool",
        "attribute_id": "attr1",
        "attribute_label": "Test Attribute",
    }
    attr = Attribute.model_validate(attr_data)

    annotation_data = {
        "attribute": attr,
        "output_data": True,
        "annotation_type": AnnotationType.HUMAN,
    }
    annotation = GoldStandardAnnotation.model_validate(annotation_data)
    assert annotation.attribute == attr
    assert annotation.output_data is True
    assert annotation.annotation_type == AnnotationType.HUMAN


def test_gold_standard_annotation_with_llm_type_from_dict() -> None:
    """Test creating annotation with LLM type from dictionary data."""
    attr_data = {
        "question_target": "Test question",
        "output_data_type": "str",
        "attribute_id": "attr2",
        "attribute_label": "Test Attribute 2",
    }
    attr = Attribute.model_validate(attr_data)

    annotation_data = {
        "attribute": attr,
        "output_data": "Test response",
        "annotation_type": AnnotationType.LLM,
    }
    annotation = GoldStandardAnnotation.model_validate(annotation_data)
    assert annotation.annotation_type == AnnotationType.LLM


def test_gold_standard_annotated_document_creation() -> None:
    """Test creating a gold standard annotated document."""
    citation = Reference(
        id=uuid4(),
        title="Test Document 3",
        authors=["Test Author 3"],
    )

    attr = Attribute(
        question_target="Test question",
        output_data_type=bool,
        attribute_id="attr3",
        attribute_label="Test Attribute 3",
    )

    annotation = GoldStandardAnnotation(
        attribute=attr,
        output_data=True,
        annotation_type=AnnotationType.HUMAN,
    )

    doc = GoldStandardAnnotatedDocument(
        name="Test Document 3",
        citation=citation,
        context="Test content",
        document_id="doc3",
        annotations=[annotation],
    )
    assert doc.name == "Test Document 3"
    assert len(doc.annotations) == 1
    assert doc.annotations[0].output_data is True
