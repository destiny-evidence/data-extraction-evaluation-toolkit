"""Tests for eppi_annotation_converter using real EPPI data."""

import json
from unittest.mock import mock_open, patch

import pytest

from deet.data_models.base import AttributeType
from deet.data_models.eppi import EppiRawData
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter

# from pytest_mock import mock_open, patch


def test_load_eppi_json_annotations(sample_eppi_data: dict) -> None:
    """Test loading EPPI JSON annotations."""
    converter = EppiAnnotationConverter()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))):
        result = converter.load_eppi_json_annotations("fake_path.json")
        assert result == sample_eppi_data
        assert "CodeSets" in result
        assert "References" in result
        assert len(result["CodeSets"]) == 2
        assert len(result["References"]) > 0


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


def test_convert_to_eppi_attributes(sample_eppi_data: dict) -> None:
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
    assert first_attr.question_target == "", "Should be empty for EPPI"


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
        assert attr.question_target == ""

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
        attribute.attribute_type == raw.get("AttributeType")
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

    attr_data_with_nulls = {
        "AttributeId": 12345,
        "AttributeName": "Test Attribute",
        "AttributeDescription": None,
        "AttributeSetDescription": None,
        "AttributeType": None,
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
    assert attr.attribute_type is None
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


def test_process_document_data_for_validation(
    sample_eppi_data: dict,
) -> None:
    """Test processing document data."""
    converter = EppiAnnotationConverter()
    first_ref = sample_eppi_data["References"][0]

    processed = converter.process_document_data_for_validation(first_ref)

    assert "name" in processed
    assert "citation" in processed
    assert "context" in processed
    assert "document_id" in processed


def test_process_attribute_data_for_validation(
    sample_eppi_data: dict,
) -> None:
    """Test processing attribute data."""
    converter = EppiAnnotationConverter()
    # Get first attribute
    first_attr = sample_eppi_data["CodeSets"][0]["Attributes"]["AttributesList"][0]

    processed = converter.process_attribute_data_for_validation(first_attr)

    assert "question_target" in processed
    assert "output_data_type" in processed
    assert "attribute_id" in processed
    assert "attribute_label" in processed
    assert processed["question_target"] == ""  # Should be empty for EPPI
    assert (
        processed["output_data_type"] == AttributeType.BOOL.value
    )  # Should be bool for EPPI


def test_create_reference_from_document_data(
    sample_eppi_data: dict,
) -> None:
    """Test creating reference from document data."""
    converter = EppiAnnotationConverter()
    first_ref = sample_eppi_data["References"][0]

    reference = converter._create_reference(first_ref)

    # Verify reference was created
    assert hasattr(reference, "id")
    assert hasattr(reference, "visibility")
    # Reference object has different structure than expected
    assert reference.id is not None


def test_integration_full_workflow(sample_eppi_data: dict) -> None:
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

        # NOTE: This fixture does not include the data needed for
        # `process_annotation_file()` to attach annotations to documents.
        # Attribute-label resolution is tested separately.


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

    # Create a minimal doc; convert_to_eppi_annotations doesn't actually use it.
    first_ref = sample_eppi_data["References"][0]
    document = converter.convert_to_eppi_document(first_ref)

    raw_codes = first_ref["Codes"]
    annotations = converter.convert_to_eppi_annotations(
        raw_codes,
        document,
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
