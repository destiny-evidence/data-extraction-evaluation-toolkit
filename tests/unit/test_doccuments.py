"""Tests for deet/data_models/"""

from destiny_sdk.references import ReferenceFileInput

from deet.data_models.documents import (
    ContextType,
    Document,
    DocumentIDSource,
    GoldStandardAnnotatedDocument,
)


def test_document_creation() -> None:
    """Test creating a document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="This is test content",
        context_type=ContextType.FULL_DOCUMENT,
        document_id=1,
        document_id_source=DocumentIDSource.EPPI_ITEM_ID,
        filename="test.pdf",
    )
    assert doc.name == "Test Document"
    assert doc.document_id == 1
    assert doc.filename == "test.pdf"
    assert doc.context == "This is test content"


def test_document_creation_with_list_context() -> None:
    """Test creating a document with list context."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document 2",
        citation=citation,
        context=["Paragraph 1", "Paragraph 2"],
        context_type=ContextType.RAG_SNIPPETS,
        document_id=2,
        document_id_source=DocumentIDSource.EPPI_ITEM_ID,
    )
    assert doc.context == ["Paragraph 1", "Paragraph 2"]


def test_gold_standard_annotated_document_creation() -> None:
    """Test creating a gold standard annotated document."""
    citation = ReferenceFileInput()

    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
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
        context_type=ContextType.FULL_DOCUMENT,
        document_id=3,
        document_id_source=DocumentIDSource.EPPI_ITEM_ID,
        annotations=[annotation],
    )
    assert doc.name == "Test Document 3"
    assert doc.document_id == 3
    assert len(doc.annotations) == 1
    assert doc.annotations[0].output_data is True
