"""Tests for annotation converter using real EPPI data."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from app.models.eppi import EppiRawData
from app.processors.annotation_converter import AnnotationConverter


class TestAnnotationConverter:
    """Test AnnotationConverter with real EPPI data."""

    @pytest.fixture
    def converter(self) -> AnnotationConverter:
        """Create AnnotationConverter instance for testing."""
        return AnnotationConverter()

    @pytest.fixture
    def sample_eppi_data(self) -> dict:
        """Load real EPPI data from test file."""
        sample_file = Path("tests/test_files/input/sample_eppi.json")
        with sample_file.open() as f:
            return json.load(f)

    def test_load_eppi_json_annotations_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test loading EPPI JSON annotations with real data."""
        with patch(
            "pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))
        ):
            result = converter.load_eppi_json_annotations("fake_path.json")
            assert result == sample_eppi_data
            assert "CodeSets" in result
            assert "References" in result
            assert len(result["CodeSets"]) == 2
            assert len(result["References"]) > 0

    def test_load_eppi_json_annotations_file_not_found(
        self, converter: AnnotationConverter
    ) -> None:
        """Test loading EPPI JSON annotations when file doesn't exist."""
        with (
            patch("pathlib.Path.open", side_effect=FileNotFoundError("File not found")),
            pytest.raises(FileNotFoundError),
        ):
            converter.load_eppi_json_annotations("nonexistent.json")

    def test_load_eppi_json_annotations_invalid_json(
        self, converter: AnnotationConverter
    ) -> None:
        """Test loading EPPI JSON annotations with invalid JSON."""
        with (
            patch("pathlib.Path.open", mock_open(read_data="invalid json")),
            pytest.raises(json.JSONDecodeError),
        ):
            converter.load_eppi_json_annotations("invalid.json")

    def test_process_annotation_file_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test processing annotation file with real EPPI data."""
        with patch(
            "pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))
        ):
            result = converter.process_annotation_file("fake_path.json")

            # Verify the result structure
            assert hasattr(result, "attributes")
            assert hasattr(result, "documents")
            assert hasattr(result, "annotations")

            # Check that attributes were processed
            assert len(result.attributes) > 0
            # Check that documents were processed
            assert len(result.documents) > 0

    def test_convert_to_eppi_attributes_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test converting to EPPI attributes with real data."""
        # Create EppiRawData from real data
        raw_data = EppiRawData.model_validate(sample_eppi_data)

        # Extract attributes from real data
        all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)  # noqa: SLF001

        # Convert to EPPI attributes
        attributes = converter.convert_to_eppi_attributes(all_attributes_raw)

        # Verify attributes were created
        assert len(attributes) > 0

        # Check first attribute
        first_attr = attributes[0]
        assert hasattr(first_attr, "attribute_id")
        assert hasattr(first_attr, "attribute_label")
        assert hasattr(first_attr, "output_data_type")
        assert first_attr.output_data_type is bool  # Should be bool for EPPI
        assert first_attr.question_target == ""  # Should be empty for EPPI

    def test_extract_attributes_from_codesets_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test extracting attributes from CodeSets with real data."""
        raw_data = EppiRawData.model_validate(sample_eppi_data)
        attributes_raw = converter._extract_attributes_from_codesets(raw_data)  # noqa: SLF001

        # Verify attributes were extracted
        assert len(attributes_raw) > 0

        # Check structure of first attribute
        first_attr = attributes_raw[0]
        assert "AttributeId" in first_attr
        assert "AttributeName" in first_attr
        assert "hierarchy_path" in first_attr
        assert "hierarchy_level" in first_attr

    def test_flatten_attributes_hierarchy_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test flattening attributes hierarchy with real data."""
        raw_data = EppiRawData.model_validate(sample_eppi_data)
        all_attributes_raw = converter._extract_attributes_from_codesets(raw_data)  # noqa: SLF001

        # Test flattening
        flattened = converter.flatten_attributes_hierarchy(all_attributes_raw)

        # Verify flattening worked
        assert len(flattened) > 0

        # Check that hierarchy information is preserved
        for attr in flattened:
            assert "hierarchy_path" in attr
            assert "hierarchy_level" in attr

    def test_validate_eppi_data_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test validating EPPI data with real data."""
        raw_data = EppiRawData.model_validate(sample_eppi_data)

        # Verify validation worked
        assert hasattr(raw_data, "code_sets")
        assert hasattr(raw_data, "references")
        assert len(raw_data.code_sets) == 2
        assert len(raw_data.references) > 0

    def test_validate_eppi_data_invalid_structure(
        self, converter: AnnotationConverter
    ) -> None:
        """Test validating EPPI data with invalid structure."""
        # EppiRawData has default values, so invalid data just gets defaults
        invalid_data = {"invalid": "structure"}

        result = EppiRawData.model_validate(invalid_data)

        # Should have default empty values
        assert result.code_sets == []
        assert result.references == []

    def test_process_document_data_for_validation_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test processing document data with real data."""
        # Get first reference from real data
        first_ref = sample_eppi_data["References"][0]

        processed = converter.process_document_data_for_validation(first_ref)

        # Verify processing worked
        assert "name" in processed
        assert "citation" in processed
        assert "context" in processed
        assert "document_id" in processed

    def test_process_attribute_data_for_validation_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test processing attribute data with real data."""
        # Get first attribute from real data
        first_attr = sample_eppi_data["CodeSets"][0]["Attributes"]["AttributesList"][0]

        processed = converter.process_attribute_data_for_validation(first_attr)

        # Verify processing worked
        assert "question_target" in processed
        assert "output_data_type" in processed
        assert "attribute_id" in processed
        assert "attribute_label" in processed
        assert processed["question_target"] == ""  # Should be empty for EPPI
        assert processed["output_data_type"] is bool  # Should be bool for EPPI

    def test_create_reference_from_document_data_with_real_data(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test creating reference from document data with real data."""
        # Get first reference from real data
        first_ref = sample_eppi_data["References"][0]

        reference = converter._create_reference(first_ref)  # noqa: SLF001

        # Verify reference was created
        assert hasattr(reference, "id")
        assert hasattr(reference, "visibility")
        # Reference object has different structure than expected
        assert reference.id is not None

    def test_integration_full_workflow(
        self, converter: AnnotationConverter, sample_eppi_data: dict
    ) -> None:
        """Test full integration workflow with real data."""
        with patch(
            "pathlib.Path.open", mock_open(read_data=json.dumps(sample_eppi_data))
        ):
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
            assert first_attr.output_data_type is bool

            # Verify document structure
            first_doc = result.documents[0]
            assert hasattr(first_doc, "name")
            assert hasattr(first_doc, "document_id")
            assert hasattr(first_doc, "citation")

    def test_error_handling_malformed_data(
        self, converter: AnnotationConverter
    ) -> None:
        """Test error handling with malformed data."""
        malformed_data = {
            "CodeSets": "not_a_list",  # Should be a list
            "References": [],
        }

        with (
            patch("pathlib.Path.open", mock_open(read_data=json.dumps(malformed_data))),
            pytest.raises(ValueError, match="Input should be a valid list"),
        ):
            converter.process_annotation_file("malformed.json")

    def test_empty_data_handling(self, converter: AnnotationConverter) -> None:
        """Test handling of empty data."""
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
