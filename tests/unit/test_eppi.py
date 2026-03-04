"""Tests for EPPI-specific models."""

import csv
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from destiny_sdk.references import ReferenceFileInput

from deet.data_models.base import AnnotationType, AttributeType
from deet.data_models.eppi import (
    EppiAttribute,
    EppiAttributeSelectionType,
    EppiDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
    EppiRawData,
    parse_citation_to_destiny,
)
from deet.data_models.processed_gold_standard_annotations import (
    CustomPromptPopulationMethod,
    ProcessedEppiAnnotationData,
)
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter


@pytest.fixture
def test_csv_file(tmp_path):
    """Create a test CSV file with 3 attributes."""
    csv_file = tmp_path / "prompts.csv"
    with csv_file.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute_id", "prompt"])
        writer.writeheader()
        writer.writerow({"attribute_id": "1", "prompt": "Test prompt 1"})
        writer.writerow({"attribute_id": "2", "prompt": "Test prompt 2"})
        writer.writerow({"attribute_id": "3", "prompt": "Test prompt 3"})
    return csv_file


@pytest.fixture
def processed_data():
    """Create ProcessedEppiAnnotationData with test attributes."""
    attr1 = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=1,
        attribute_label="Attribute 1",
        output_data_type=AttributeType.BOOL,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    attr2 = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=2,
        attribute_label="Attribute 2",
        output_data_type=AttributeType.BOOL,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    attr3 = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=3,
        attribute_label="Attribute 3",
        output_data_type=AttributeType.BOOL,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    return ProcessedEppiAnnotationData(
        attributes=[attr1, attr2, attr3],
        documents=[],
        annotations=[],
        annotated_documents=[],
        attribute_id_to_label={1: "Attribute 1", 2: "Attribute 2", 3: "Attribute 3"},
        raw_data=EppiRawData(),
    )


def test_eppi_attribute_creation_from_json_data() -> None:
    """Test creating EppiAttribute from JSON-like data (would come from EPPI JSON)."""
    # This mimics how EppiAttribute is created from JSON data in the ann converter
    # The data structure matches what comes from EPPI JSON files
    # The converter now properly maps camelCase fields to snake_case
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
        "output_data_type": AttributeType.BOOL.value,  # Always boolean for EPPI
        "attribute_id": "5730447",  # Converted to string
        "attribute_label": "Test EPPI Attribute",
        # Converter now maps camelCase to snake_case automatically
        "attribute_set_description": "Test set description",
        "attribute_type": "Selectable (show checkbox)",
        "attribute_description": "Test description",
    }
    attr = EppiAttribute.model_validate(attr_data)
    assert attr.attribute_id == 5730447
    assert attr.attribute_label == "Test EPPI Attribute"
    assert attr.output_data_type.to_python_type() is bool  # Default boolean for EPPI
    # Test the EPPI-specific fields are properly populated
    assert attr.attribute_set_description == "Test set description"
    assert attr.attribute_selection_type == "Selectable (show checkbox)"
    assert attr.attribute_description == "Test description"


def test_eppi_attribute_with_eppi_fields() -> None:
    """Test creating EppiAttribute with EPPI-specific fields."""
    attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=2345,  # as mypy forgets base.py inheritance
        attribute_label="Test EPPI Attribute 2",
        attribute_set_description="Test description",
        hierarchy_path="root.test.attribute",
        hierarchy_level=2,
        is_leaf=False,
        parent_attribute_id=123,
        attribute_type=EppiAttributeSelectionType.SELECTABLE,
    )
    assert attr.attribute_set_description == "Test description"
    assert attr.hierarchy_path == "root.test.attribute"
    assert attr.hierarchy_level == 2
    assert attr.is_leaf is False
    assert attr.parent_attribute_id == 123
    assert attr.attribute_selection_type == EppiAttributeSelectionType.SELECTABLE


def eppi_attribute_selection_type_case_insensitivity() -> None:
    """Test the custom case-insensitivity for our attribute selection enum."""
    outcome_a = EppiAttributeSelectionType("outcome")
    outcome_b = EppiAttributeSelectionType("OUTCOME")
    outcome_c = EppiAttributeSelectionType("ouTCoMe")
    assert outcome_a == EppiAttributeSelectionType.OUTCOME
    assert outcome_b == EppiAttributeSelectionType.OUTCOME
    assert outcome_c == EppiAttributeSelectionType.OUTCOME


def test_eppi_attribute_with_different_output_types() -> None:
    """Test EppiAttribute with different output_data_type values."""
    # Test with str type
    attr_str = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=3456,
        attribute_label="String Attribute",
        output_data_type=AttributeType.STRING,
        attribute_type=EppiAttributeSelectionType.SELECTABLE,
    )
    assert attr_str.output_data_type.value == AttributeType.STRING.value

    # Test with int type
    attr_int = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=4567,
        attribute_label="Integer Attribute",
        output_data_type=AttributeType.INTEGER,
        attribute_type=EppiAttributeSelectionType.SELECTABLE,
    )
    assert attr_int.output_data_type.value == AttributeType.INTEGER.value


def test_eppi_attribute_camel_case_mapping() -> None:
    """Test that camelCase JSON fields are mapped correctly."""
    attr_dict = {
        "AttributeId": 5678,
        "AttributeName": "Attri Bute",
        "AttributeLabel": "Test attribute",
        "AttributeSetDescription": "bla",
        "AttributeType": "Selectable (show checkbox)",
    }

    attr = EppiAttribute(**attr_dict)  # type: ignore[arg-type]
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


def test_eppi_gold_standard_annotation_creation_from_json_data() -> None:
    """Test creating EppiGoldStandardAnnotation from JSON-like data."""
    # This mimics how annotations are created from EPPI JSON data
    # First create the attribute from JSON-like data
    attr_data = {
        "AttributeId": 5730447,
        "AttributeName": "Test EPPI Attribute",
        "AttributeType": "Selectable (show checkbox)",
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
    attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=2345,
        attribute_label="Test EPPI Attribute 2",
        output_data_type=AttributeType.STRING,
        attribute_type=EppiAttributeSelectionType.SELECTABLE,
    )

    annotation = EppiGoldStandardAnnotation(
        attribute=attr,
        output_data="Test response",
        annotation_type=AnnotationType.LLM,
    )
    assert annotation.annotation_type == AnnotationType.LLM


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


def test_import_prompts_csv_updates_output_data_type(
    sample_eppi_data: dict, tmp_path: Path
) -> None:
    """Test that populate_custom_prompts from CSV updates output_data_type."""
    data_no_codes = {**sample_eppi_data}
    for ref in data_no_codes.get("References", []):
        ref["Codes"] = []

    converter = EppiAnnotationConverter()
    with patch(
        "pathlib.Path.open",
        mock_open(read_data=json.dumps(data_no_codes)),
    ):
        result = converter.process_annotation_file(tmp_path / "fake.json")

    # Should have attributes with default output_data_type=BOOL
    assert len(result.attributes) > 0
    first_attr = result.attributes[0]
    assert all(
        [attr.output_data_type == AttributeType.BOOL for attr in result.attributes]  # noqa: C419
    )

    # Create CSV with output_data_type=string for first attribute
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text(
        "attribute_id,prompt,output_data_type\n"
        f"{first_attr.attribute_id},Test prompt,string\n",
        encoding="utf-8",
    )

    result.populate_custom_prompts(
        method=CustomPromptPopulationMethod.FILE, filepath=csv_path
    )

    assert first_attr.output_data_type == AttributeType.STRING
    assert result.raw_data is not None
    assert len(result.raw_data.code_sets[0].attributes["AttributesList"]) == 1  # type:ignore[index]


# minimal tests for csv import
def test_import_prompts_csv_file_comprehensive(test_csv_file, processed_data) -> None:
    """Test CSV import functionality."""
    processed_data._import_prompts_csv_file(test_csv_file)

    assert len(processed_data.attributes) == 3

    assert processed_data.attributes[0].prompt == "Test prompt 1"
    assert processed_data.attributes[1].prompt == "Test prompt 2"
    assert processed_data.attributes[2].prompt == "Test prompt 3"


def test_import_prompts_with_csv_missing_prompt(test_csv_file, processed_data) -> None:
    """Test csv with one row missing prompts."""
    with test_csv_file.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute_id", "prompt"])
        writer.writerow({"attribute_id": "4", "prompt": ""})

    processed_data._import_prompts_csv_file(test_csv_file)
    # we should now have 3 attributes in processed_data as
    # attribute_id==4 is missing a prompt.
    assert len(processed_data.attributes) == 3

    assert processed_data.attributes[0].prompt == "Test prompt 1"
    assert processed_data.attributes[1].prompt == "Test prompt 2"
    assert processed_data.attributes[2].prompt == "Test prompt 3"


def test_import_prompts_with_csv_missing_prompt_read_all_attributes(
    test_csv_file, processed_data
) -> None:
    """Test csv with one row missing prompts and retain_only_csv_attributes=False."""
    with test_csv_file.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attribute_id", "prompt"])
        writer.writerow({"attribute_id": "4", "prompt": ""})

    attr4 = EppiAttribute(  # type:ignore[call-arg]
        attribute_id=4,
        output_data_type=AttributeType.BOOL,
        attribute_selection_type=EppiAttributeSelectionType.INTERVENTION,
        attribute_label="yes",
    )
    processed_data.attributes.append(attr4)

    # note the added arg - this means we keep all original attributes
    processed_data._import_prompts_csv_file(
        test_csv_file, retain_only_csv_attributes=False
    )
    assert len(processed_data.attributes) == 4

    assert processed_data.attributes[0].prompt == "Test prompt 1"
    assert processed_data.attributes[1].prompt == "Test prompt 2"
    assert processed_data.attributes[2].prompt == "Test prompt 3"
    # attr4's prompt was not updated because CSV had empty prompt for ID 4
    assert processed_data.attributes[3].prompt is None


# eppi document citation generation
@pytest.mark.parametrize(
    ("date_input", "expected_day", "expected_month", "expected_year"),
    [
        ("15/03/2024", 15, 3, 2024),
        ("2024-03-15 10:30:00+0000", 15, 3, 2024),
        ("2024-03-15", 15, 3, 2024),
        ("01/12/2023", 1, 12, 2023),
        ("31/01/2020", 31, 1, 2020),
    ],
    ids=[
        "dd_mm_yyyy_format",
        "iso_format_with_timezone",
        "simple_iso_format",
        "first_day_of_month",
        "last_day_of_month",
    ],
)
def test_parse_date_string_valid_formats(
    date_input: str,
    expected_day: int,
    expected_month: int,
    expected_year: int,
):
    """Test parsing various valid date formats."""
    doc = EppiDocument(
        citation=ReferenceFileInput(),
        document_id=123,
        name="Test Doc",
        date_created=date_input,  # type:ignore[arg-type]
    )
    assert doc.date_created is not None
    assert isinstance(doc.date_created, datetime)
    assert doc.date_created.day == expected_day
    assert doc.date_created.month == expected_month
    assert doc.date_created.year == expected_year


def test_parse_date_string_none_value():
    """Test that None date_created remains None."""
    doc = EppiDocument(
        document_id=123,
        name="Test Doc",
        date_created=None,
        citation=ReferenceFileInput(),
    )
    assert doc.date_created is None


def test_parse_date_string_empty_string():
    """Test that empty string date_created becomes None."""
    doc = EppiDocument(
        document_id=123,
        name="Test Doc",
        date_created="",  # type:ignore[arg-type]
        citation=ReferenceFileInput(),
    )
    assert doc.date_created is None


def test_parse_date_string_invalid_format_raises():
    """Test that invalid date format raises ValueError."""
    with pytest.raises(ValueError, match="unable to parse date_created"):
        EppiDocument(
            citation=ReferenceFileInput(),
            document_id=123,
            name="Test Doc",
            date_created="not-a-date",  # type:ignore[arg-type]
        )


# destiny citation validator
def test_populate_citation_field_from_reference(sample_eppi_data):
    """Test that citation field is populated from EPPI reference data."""
    reference = sample_eppi_data["References"][0]
    doc = EppiDocument.model_validate(reference)

    assert doc.citation is not None
    assert isinstance(doc.citation, ReferenceFileInput)
    assert doc.document_id == 28856292
    assert doc.name == "A title"


def test_populate_citation_field_preserves_existing():
    """Test that existing citation field is not overwritten."""
    existing_citation = parse_citation_to_destiny({"abstract": "abstract"})

    doc = EppiDocument(
        document_id=123,
        name="Test Doc",
        citation=existing_citation,
    )
    assert doc.citation is existing_citation


def test_populate_citation_with_doi():
    """Test citation population with DOI field."""
    reference_data = {
        "ItemId": 12345,
        "Title": "Test Article",
        "Year": "2023",
        "Authors": "Doe, J;",
        "DOI": "10.1234/test.doi.5678",
        "Abstract": "Test abstract",
    }
    doc = EppiDocument.model_validate(reference_data)

    assert doc.citation is not None
    assert isinstance(doc.citation, ReferenceFileInput)
    assert doc.doi == "10.1234/test.doi.5678"


def test_populate_citation_with_malformed_doi():
    """Test citation population sanitises malformed DOI."""
    reference_data = {
        "ItemId": 12345,
        "Title": "Test Article",
        "DOI": "https://doi.org/10.1234/test.doi.5678",
    }
    doc = EppiDocument.model_validate(reference_data)

    # doi should be sanitised
    assert doc.doi == "10.1234/test.doi.5678"
