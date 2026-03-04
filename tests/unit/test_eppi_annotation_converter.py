"""Tests for eppi_annotation_converter using real EPPI data."""

import csv
import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from pytest_mock import MockerFixture

from deet.data_models.base import AttributeType
from deet.data_models.eppi import (
    EppiAttributeSelectionType,
    EppiDocument,
    EppiRawData,
)
from deet.data_models.processed import ProcessedEppiAnnotationData
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter


def test_load_eppi_json_annotations(
    sample_eppi_data: dict, mocker: MockerFixture
) -> None:
    """Test loading EPPI JSON annotations via process_annotation_file."""
    converter = EppiAnnotationConverter()
    mocker.patch.object(
        Path, "open", mocker.mock_open(read_data=json.dumps(sample_eppi_data))
    )
    result = converter.process_annotation_file("fake_path.json")
    assert hasattr(result, "raw_data")
    assert len(result.raw_data.code_sets) == 2
    assert len(result.raw_data.references) > 0


def test_process_annotation_file_with_real_data(sample_eppi_data: dict) -> None:
    """Test processing annotation file with real EPPI data."""
    converter = EppiAnnotationConverter()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))):
        result = converter.process_annotation_file("fake_path.json")

        assert hasattr(result, "attributes")
        assert hasattr(result, "documents")
        assert hasattr(result, "annotations")

        assert len(result.attributes) > 0
        assert len(result.documents) > 0


@pytest.fixture
def processed_eppi_annotations(sample_eppi_data: dict) -> ProcessedEppiAnnotationData:
    """Create fixture to test methods that operate on ProcessedEppiAnnotationData."""
    converter = EppiAnnotationConverter()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))):
        return converter.process_annotation_file("fake_path.json")


@pytest.fixture
def attribute_csv(tmp_path, processed_eppi_annotations):
    """Create fixture to write csv file from processed eppi annotation data."""
    csv_file = tmp_path / "attribute_definitions.csv"
    processed_eppi_annotations.export_attributes_csv_file(csv_file)
    return csv_file


def test_export_attributes_csv(attribute_csv, processed_eppi_annotations):
    """
    Test whether csv has been written, has the expected number of rows, and
    has at an attribute_id and prompt field in the headers.
    """
    with attribute_csv.open() as f:
        reader = csv.reader(f)
        rows = list(reader)

    headers = rows[0]
    assert "attribute_id" in headers
    assert "prompt" in headers

    assert len(rows) - 1 == len(processed_eppi_annotations.attributes)


@pytest.fixture
def edited_attribute_csv(tmp_path, attribute_csv):
    """
    Edit the csv, putting a new prompt in the prompt column.
    Write it to a new fixture.
    """
    edited_definitions = tmp_path / "edited_attribute_definitions.csv"
    with attribute_csv.open() as fr, edited_definitions.open("w") as fw:
        reader = csv.reader(fr)
        writer = csv.writer(fw)
        for i, row in enumerate(reader):
            if i == 0:
                writer.writerow(row)
            else:
                row[0] = "New edited prompt"
                writer.writerow(row)
    return edited_definitions


def test_populate_custom_prompts_csv(edited_attribute_csv, processed_eppi_annotations):
    """
    Test whether populating custom prompts from the edited csv correctly sets
    the prompt for each attribute.
    """
    processed_eppi_annotations.populate_custom_prompts(
        method="file", filepath=edited_attribute_csv
    )
    for attribute in processed_eppi_annotations.attributes:
        assert attribute.prompt == "New edited prompt"


def test_populate_custom_prompts_cli(processed_eppi_annotations):
    """Test the CLI form populating custom prompts."""
    side_effect_list = []
    for _ in processed_eppi_annotations.attributes:
        side_effect_list.extend(["y", "New edited prompt", "y"])
    with patch("builtins.input", side_effect=side_effect_list):
        processed_eppi_annotations.populate_custom_prompts(method="cli")
    for attribute in processed_eppi_annotations.attributes:
        assert attribute.prompt == "New edited prompt"


def test_convert_to_eppi_attributes_default(sample_eppi_data: dict) -> None:
    """Test converting to EPPI attributes."""
    converter = EppiAnnotationConverter()
    raw_data = EppiRawData.model_validate(sample_eppi_data)

    all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)

    attributes = converter.convert_to_eppi_attributes(all_attributes_raw)

    assert len(attributes) > 0

    first_attr = attributes[0]
    assert hasattr(first_attr, "attribute_id")
    assert hasattr(first_attr, "attribute_label")
    assert hasattr(first_attr, "output_data_type")
    assert (
        first_attr.output_data_type == AttributeType.BOOL.value
    ), "Should be bool for EPPI"


def test_convert_to_eppi_attributes_custom_attribute_type(sample_eppi_data) -> None:
    """Test converting to EPPI attributes /w custom attribute type."""
    converter = EppiAnnotationConverter()
    raw_data = EppiRawData.model_validate(sample_eppi_data)

    all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)

    attributes = converter.convert_to_eppi_attributes(
        all_attributes_raw, set_attribute_type=AttributeType.STRING
    )
    assert len(attributes) > 0
    assert all(hasattr(attr, "attribute_id") for attr in attributes)
    assert all(hasattr(attr, "attribute_label") for attr in attributes)
    assert all(hasattr(attr, "output_data_type") for attr in attributes)
    assert all(attr.output_data_type == AttributeType.STRING for attr in attributes)


def test_convert_to_eppi_attributes_field_population(
    sample_eppi_data: dict,
) -> None:
    """Test that all fields are properly populated when converting attributes."""
    converter = EppiAnnotationConverter()
    raw_data = EppiRawData.model_validate(sample_eppi_data)

    all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)

    attributes = converter.convert_to_eppi_attributes(all_attributes_raw)

    assert len(attributes) > 0

    # Check that all expected fields are populated
    for attr in attributes:
        # Core fields
        assert attr.attribute_id is not None
        assert attr.attribute_label is not None
        assert attr.output_data_type == AttributeType.BOOL.value

        # EPPI-specific fields should be populated
        # (not None unless explicitly None in JSON)
        # attribute_type should be populated if present in JSON
        # attribute_description should be populated if present in JSON
        # attribute_set_description should be populated if present in JSON
        # hierarchy_path should be a string (may be empty for root level)
        assert isinstance(attr.hierarchy_path, str)
        assert isinstance(attr.hierarchy_level, int)
        assert isinstance(attr.is_leaf, bool)

    assert all(
        attribute.attribute_selection_type == raw.get("AttributeType")
        for attribute, raw in zip(attributes, all_attributes_raw, strict=False)
        if raw.get("AttributeType") is not None
    ), "attribute_type should match for all attributes where present"
    assert all(
        attribute.attribute_description == raw.get("AttributeDescription")
        for attribute, raw in zip(attributes, all_attributes_raw, strict=False)
        if "AttributeDescription" in raw
    ), "attribute_description should match for all attributes where present"
    assert all(
        attribute.attribute_set_description == raw.get("AttributeSetDescription")
        for attribute, raw in zip(attributes, all_attributes_raw, strict=False)
        if raw.get("AttributeSetDescription") is not None
    ), "attribute_set_description should match for all attributes where present"


def test_convert_to_eppi_attributes_with_null_values() -> None:
    """Test that null/None values in JSON are handled for EPPI-specific fields."""
    converter = EppiAnnotationConverter()

    # AttributeType is required, so we must provide a valid value
    attr_data_with_nulls = {
        "AttributeId": 12345,
        "AttributeName": "Test Attribute",
        "AttributeDescription": None,
        "AttributeSetDescription": None,
        "AttributeType": EppiAttributeSelectionType.SELECTABLE.value,
        "hierarchy_path": "",
        "hierarchy_level": 0,
        "is_leaf": True,
    }

    attributes = converter.convert_to_eppi_attributes([attr_data_with_nulls])

    assert len(attributes) == 1
    attr = attributes[0]

    assert attr.attribute_id == 12345
    assert attr.attribute_label == "Test Attribute"
    assert attr.attribute_description is None
    assert attr.attribute_set_description is None
    assert attr.attribute_selection_type == EppiAttributeSelectionType.SELECTABLE
    assert attr.hierarchy_path == ""
    assert attr.hierarchy_level == 0
    assert attr.is_leaf is True


def test_extract_attributes_from_codesets(
    sample_eppi_data: dict,
) -> None:
    """Test extracting attributes from CodeSets."""
    converter = EppiAnnotationConverter()
    raw_data = EppiRawData.model_validate(sample_eppi_data)
    attributes_raw = converter._extract_attributes_from_codesets(raw_data)

    # Verify attributes were extracted
    assert len(attributes_raw) > 0

    # Check structure of first attribute
    first_attr = attributes_raw[0]
    assert "AttributeId" in first_attr
    assert "AttributeName" in first_attr
    assert "hierarchy_path" in first_attr
    assert "hierarchy_level" in first_attr


def test_flatten_attributes_hierarchy(sample_eppi_data: dict) -> None:
    """Test flattening attributes hierarchy with real data."""
    converter = EppiAnnotationConverter()
    raw_data = EppiRawData.model_validate(sample_eppi_data)
    all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)

    flattened = converter.flatten_attributes_hierarchy(all_attributes_raw)

    assert len(flattened) > 0

    for attr in flattened:
        assert "hierarchy_path" in attr
        assert "hierarchy_level" in attr


def test_validate_eppi_data(sample_eppi_data: dict) -> None:
    """Test validating EPPI data."""
    raw_data = EppiRawData.model_validate(sample_eppi_data)

    assert hasattr(raw_data, "code_sets")
    assert hasattr(raw_data, "references")
    assert len(raw_data.code_sets) == 2
    assert len(raw_data.references) > 0


def test_validate_eppi_data_invalid_structure() -> None:
    """Test validating EPPI data with invalid structure."""
    # EppiRawData has default values, so invalid data just gets defaults
    invalid_data = {"invalid": "structure"}

    result = EppiRawData.model_validate(invalid_data)

    # Should have default empty values
    assert result.code_sets == []
    assert result.references == []


def test_process_document_data(
    sample_eppi_data: dict,
) -> None:
    """Test processing document data by creating an EppiDocument."""
    first_ref = sample_eppi_data["References"][0]

    # EppiDocument handles the conversion directly
    doc = EppiDocument(**first_ref)

    assert doc.name is not None
    assert doc.citation is not None
    assert doc.document_id is not None


def test_process_attribute_data(
    sample_eppi_data: dict,
) -> None:
    """Test processing attribute data via convert_to_eppi_attributes."""
    converter = EppiAnnotationConverter()
    first_attr = sample_eppi_data["CodeSets"][0]["Attributes"]["AttributesList"][0]

    first_attr["hierarchy_path"] = ""
    first_attr["hierarchy_level"] = 0
    first_attr["is_leaf"] = True

    attributes = converter.convert_to_eppi_attributes([first_attr])

    assert len(attributes) == 1
    attr = attributes[0]
    assert attr.output_data_type == AttributeType.BOOL  # bool for EPPI


def test_create_document_from_reference_data(
    sample_eppi_data: dict,
) -> None:
    """Test creating EppiDocument from reference data."""
    first_ref = sample_eppi_data["References"][0]

    # EppiDocument creates the reference via model_validator
    doc = EppiDocument(**first_ref)

    # Verify document was created
    assert doc.citation is not None


def test_full_workflow(sample_eppi_data: dict) -> None:
    """Test full integration workflow."""
    converter = EppiAnnotationConverter()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))):
        result = converter.process_annotation_file("fake_path.json")

        # Verify complete workflow
        assert hasattr(result, "attributes")
        assert hasattr(result, "documents")
        assert hasattr(result, "annotations")

        # Check that we have processed data
        assert len(result.attributes) > 0
        assert len(result.documents) > 0

        # Verify attribute structure
        first_attr = result.attributes[0]
        assert hasattr(first_attr, "attribute_id")
        assert hasattr(first_attr, "attribute_label")
        assert hasattr(first_attr, "output_data_type")
        assert first_attr.output_data_type == AttributeType.BOOL.value

        # Verify document structure
        first_doc = result.documents[0]
        assert hasattr(first_doc, "name")
        assert hasattr(first_doc, "document_id")
        assert hasattr(first_doc, "citation")


@pytest.mark.parametrize(
    ("attribute_type_str", "expected_type"),
    [
        ("string", AttributeType.STRING),
        ("integer", AttributeType.INTEGER),
        ("bool", AttributeType.BOOL),
        ("list", AttributeType.LIST),
        ("dict", AttributeType.DICT),
    ],
)
def test_full_workflow_custom_att_type_str_valid(
    sample_eppi_data, attribute_type_str: str, expected_type: AttributeType
) -> None:
    """Testing full workflow with valid custom att type as string."""
    converter = EppiAnnotationConverter()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))):
        result = converter.process_annotation_file(
            "fake_path.json", set_attribute_type=attribute_type_str
        )
        first_attr = result.attributes[0]
        assert first_attr.output_data_type == expected_type


@pytest.mark.parametrize(
    "attribute_type_str",
    [
        "STRING",
        "custom",
        "foo",
        "!bar",
    ],
)
def test_full_workflow_custom_att_type_str_invalid(
    sample_eppi_data,
    attribute_type_str: str,
) -> None:
    """Testing full workflow with invalid custom att type as string."""
    converter = EppiAnnotationConverter()
    with (
        patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))),
        pytest.raises(ValueError, match="is not a valid AttributeType"),
    ):
        converter.process_annotation_file(
            "fake_path.json", set_attribute_type=attribute_type_str
        )


def test_convert_to_eppi_annotations_uses_codeset_attribute_label(
    sample_eppi_data: dict,
) -> None:
    """Regression test for #93: avoid fallback `Attribute <id>` labels."""
    converter = EppiAnnotationConverter()

    raw_data = EppiRawData.model_validate(sample_eppi_data)
    all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)
    attributes = converter.convert_to_eppi_attributes(all_attributes_raw)

    attributes_lookup = {attr.attribute_id: attr for attr in attributes}
    attribute_id_to_label = {
        attr.attribute_id: attr.attribute_label for attr in attributes
    }

    first_ref = sample_eppi_data["References"][0]
    raw_codes = first_ref["Codes"]

    annotations = converter.convert_to_eppi_annotations(
        raw_codes,
        attributes_lookup=attributes_lookup,
        attribute_id_to_label=attribute_id_to_label,
    )

    assert len(annotations) == 1
    assert all(
        annotation.attribute.attribute_id == attribute_id
        for annotation, attribute_id in zip(
            annotations, attributes_lookup.keys(), strict=False
        )
    )
    assert all(
        annotation.attribute.attribute_label
        == attributes_lookup[attribute_id].attribute_label
        for annotation, attribute_id in zip(
            annotations, attributes_lookup.keys(), strict=False
        )
    )


def test_error_handling_malformed_data() -> None:
    """Test error handling with malformed data."""
    converter = EppiAnnotationConverter()
    malformed_data = {
        "CodeSets": "not_a_list",  # Should be a list
        "References": [],
    }

    with (
        patch("pathlib.Path.open", mock_open(read_data=json.dumps(malformed_data))),
        pytest.raises(ValueError, match="Input should be a valid list"),
    ):
        converter.process_annotation_file("malformed.json")


def test_empty_data_handling() -> None:
    """Test handling of empty data."""
    converter = EppiAnnotationConverter()
    empty_data: dict = {
        "CodeSets": [],
        "References": [],
    }

    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(empty_data))):
        result = converter.process_annotation_file("empty.json")

        # Should handle empty data gracefully
        assert hasattr(result, "attributes")
        assert hasattr(result, "documents")
        assert hasattr(result, "annotations")
        assert len(result.attributes) == 0
        assert len(result.documents) == 0
        assert len(result.attributes) == 0
        assert len(result.documents) == 0
