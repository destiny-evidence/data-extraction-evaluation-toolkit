"""Tests for core base models."""

from app.data_models.base import (
    AnnotationType,
    Attribute,
    AttributesList,
    GoldStandardAnnotation,
)


class TestAttribute:
    """Test Attribute model with new union type for output_data_type."""

    def test_attribute_creation_with_bool_type(self) -> None:
        """Test creating attribute with bool type."""
        attr = Attribute(
            question_target="Is this a test?",
            output_data_type=bool,
            attribute_id="test1",
            attribute_label="Test Boolean Attribute",
        )
        assert attr.question_target == "Is this a test?"
        assert attr.output_data_type is bool
        assert attr.attribute_id == "test1"
        assert attr.attribute_label == "Test Boolean Attribute"

    def test_attribute_creation_with_str_type(self) -> None:
        """Test creating attribute with str type."""
        attr = Attribute(
            question_target="What is the name?",
            output_data_type=str,
            attribute_id="test2",
            attribute_label="Test String Attribute",
        )
        assert attr.output_data_type is str

    def test_attribute_creation_with_int_type(self) -> None:
        """Test creating attribute with int type."""
        attr = Attribute(
            question_target="How many items?",
            output_data_type=int,
            attribute_id="test3",
            attribute_label="Test Integer Attribute",
        )
        assert attr.output_data_type is int

    def test_attribute_creation_with_list_type(self) -> None:
        """Test creating attribute with list type."""
        attr = Attribute(
            question_target="What are the items?",
            output_data_type=list,
            attribute_id="test4",
            attribute_label="Test List Attribute",
        )
        assert attr.output_data_type is list

    def test_attribute_creation_with_dict_type(self) -> None:
        """Test creating attribute with dict type."""
        attr = Attribute(
            question_target="What are the details?",
            output_data_type=dict,
            attribute_id="test5",
            attribute_label="Test Dictionary Attribute",
        )
        assert attr.output_data_type is dict

    def test_attribute_creation_with_float_type(self) -> None:
        """Test creating attribute with float type."""
        attr = Attribute(
            question_target="What is the value?",
            output_data_type=float,
            attribute_id="test6",
            attribute_label="Test Float Attribute",
        )
        assert attr.output_data_type is float

    def test_attribute_validation_required_fields(self) -> None:
        """Test that required fields are validated."""
        # Test that we can create attributes with valid data
        attr = Attribute(
            question_target="Test",
            output_data_type=bool,
            attribute_id="test_id",
            attribute_label="Test Label",
        )
        assert attr.question_target == "Test"
        assert attr.attribute_id == "test_id"
        assert attr.attribute_label == "Test Label"


class TestAttributesList:
    """Test AttributesList container."""

    def test_attributes_list_creation(self) -> None:
        """Test creating AttributesList with multiple attributes."""
        attrs = [
            Attribute(
                question_target="Question 1",
                output_data_type=bool,
                attribute_id="attr1",
                attribute_label="Attribute 1",
            ),
            Attribute(
                question_target="Question 2",
                output_data_type=str,
                attribute_id="attr2",
                attribute_label="Attribute 2",
            ),
        ]
        attr_list = AttributesList(attributes=attrs)
        assert len(attr_list.attributes) == 2
        assert attr_list.attributes[0].attribute_id == "attr1"
        assert attr_list.attributes[1].attribute_id == "attr2"

    def test_attributes_list_iteration(self) -> None:
        """Test that AttributesList is iterable."""
        attrs = [
            Attribute(
                question_target="Question 1",
                output_data_type=bool,
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


class TestDocument:
    """Test Document model."""

    # def test_document_creation(self) -> None:
    #     """Test creating a document."""
    #     citation = Reference(
    #         id=uuid4(),
    #         title="Test Document",
    #         authors=["Test Author"],
    #     )
    #     doc = Document(
    #         name="Test Document",
    #         citation=citation,
    #         context="This is test content",
    #         document_id="doc1",
    #         filename="test.pdf",
    #     )
    #     assert doc.name == "Test Document"
    #     assert doc.document_id == "doc1"
    #     assert doc.filename == "test.pdf"
    #     assert doc.context == "This is test content"

    # def test_document_creation_with_list_context(self) -> None:
    #     """Test creating a document with list context."""
    #     citation = Reference(
    #         id=uuid4(),
    #         title="Test Document 2",
    #         authors=["Test Author 2"],
    #     )
    #     doc = Document(
    #         name="Test Document 2",
    #         citation=citation,
    #         context=["Paragraph 1", "Paragraph 2"],
    #         document_id="doc2",
    #     )
    #     assert doc.context == ["Paragraph 1", "Paragraph 2"]


class TestGoldStandardAnnotation:
    """Test GoldStandardAnnotation model."""

    def test_gold_standard_annotation_creation(self) -> None:
        """Test creating a gold standard annotation."""
        attr = Attribute(
            question_target="Test question",
            output_data_type=bool,
            attribute_id="attr1",
            attribute_label="Test Attribute",
        )

        annotation = GoldStandardAnnotation(
            attribute=attr,
            output_data=True,
            annotation_type=AnnotationType.HUMAN,
        )
        assert annotation.attribute == attr
        assert annotation.output_data is True
        assert annotation.annotation_type == AnnotationType.HUMAN

    def test_gold_standard_annotation_with_llm_type(self) -> None:
        """Test creating annotation with LLM type."""
        attr = Attribute(
            question_target="Test question",
            output_data_type=str,
            attribute_id="attr2",
            attribute_label="Test Attribute 2",
        )

        annotation = GoldStandardAnnotation(
            attribute=attr,
            output_data="Test response",
            annotation_type=AnnotationType.LLM,
        )
        assert annotation.annotation_type == AnnotationType.LLM


# class TestGoldStandardAnnotatedDocument:
#     """Test GoldStandardAnnotatedDocument model."""

#     def test_gold_standard_annotated_document_creation(self) -> None:
#         """Test creating a gold standard annotated document."""
#         citation = Reference(
#             id=uuid4(),
#             title="Test Document 3",
#             authors=["Test Author 3"],
#         )

#         attr = Attribute(
#             question_target="Test question",
#             output_data_type=bool,
#             attribute_id="attr3",
#             attribute_label="Test Attribute 3",
#         )

#         annotation = GoldStandardAnnotation(
#             attribute=attr,
#             output_data=True,
#             annotation_type=AnnotationType.HUMAN,
#         )

#         doc = GoldStandardAnnotatedDocument(
#             name="Test Document 3",
#             citation=citation,
#             context="Test content",
#             document_id="doc3",
#             annotations=[annotation],
#         )
#         assert doc.name == "Test Document 3"
#         assert len(doc.annotations) == 1
#         assert doc.annotations[0].output_data is True
