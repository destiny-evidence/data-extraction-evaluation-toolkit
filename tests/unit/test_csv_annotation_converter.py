"""Tests for csv_annotation_converter."""

import io
from pathlib import Path

import pytest

from deet.data_models.base import AttributeType
from deet.processors.csv_annotation_converter import (
    ColumnTypeInferenceError,
    CSVAnnotationConverter,
)


@pytest.fixture
def converter():
    return CSVAnnotationConverter(base_output_dir=None)


# --- _infer_type ---
class TestInferType:
    """Tests for _infer_type."""

    def test_bool(self, converter):
        """Bool strings inferred as bool."""
        assert converter._infer_type("true") is bool
        assert converter._infer_type("False") is bool
        assert converter._infer_type("t") is bool

    def test_int(self, converter):
        """Integer strings inferred as int."""
        assert converter._infer_type("42") is int
        assert converter._infer_type("-3") is int

    def test_float(self, converter):
        """Float strings inferred as float."""
        assert converter._infer_type("3.14") is float

    def test_str(self, converter):
        """Non-numeric strings inferred as str."""
        assert converter._infer_type("hello") is str

    def test_empty_returns_none(self, converter):
        """Empty or whitespace-only strings return None."""
        assert converter._infer_type("") is None
        assert converter._infer_type("   ") is None


# --- _resolve_types ---
class TestResolveTypes:
    """Tests for _resolve_types."""

    def test_all_int(self, converter):
        """All int types resolve to INTEGER."""
        assert converter._resolve_types([int, None, int]) == AttributeType.INTEGER

    def test_int_and_float_resolves_float(self, converter):
        """Mixed int and float resolves to FLOAT."""
        assert converter._resolve_types([int, float]) == AttributeType.FLOAT

    def test_all_str(self, converter):
        """All str types resolve to STRING."""
        assert converter._resolve_types([str, str]) == AttributeType.STRING

    def test_all_none_raises(self, converter):
        """All-null sample raises ValueError."""
        with pytest.raises(ValueError, match="null"):
            converter._resolve_types([None, None])

    def test_incompatible_types_raises(self, converter):
        """Incompatible types raise ColumnTypeInferenceError."""
        with pytest.raises(ColumnTypeInferenceError):
            converter._resolve_types([str, int])


# --- _find_duplicate_column_names ---
class TestFindDuplicateColumnNames:
    """Tests for _find_duplicate_column_names."""

    def test_no_duplicates(self, converter):
        """Returns empty list when no duplicates."""
        assert converter._find_duplicate_column_names(["a", "b", "c"]) == []

    def test_finds_duplicate(self, converter):
        """Returns duplicate names and their positions."""
        result = converter._find_duplicate_column_names(["a", "b", "a"])
        assert result == [{"a": [0, 2]}]


class TestLoadCsv:
    """Tests for load_csv."""

    @pytest.fixture
    def csv_content(self):
        """Minimal valid CSV string for load_csv tests."""
        return (
            "name,document_id,num_patients,about_adaptation\n"
            "Paper A,1,42,t\n"
            "Paper B,2,70,f\n"
        )

    def test_loads_valid_csv(self, converter, csv_content, mocker):
        """Valid CSV loads with correct columns and row count."""
        mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
        colnames, ref_fields, attr_fields, rows = converter.load_csv("fake.csv")
        assert "name" in colnames
        assert "document_id" in colnames
        assert len(rows) == 2

    def test_missing_required_column_raises(self, converter, mocker):
        """Missing required column raises ValueError."""
        mocker.patch.object(
            Path, "open", return_value=io.StringIO("name,num_patients\nPaper A,10\n")
        )
        with pytest.raises(ValueError, match="document_id"):
            converter.load_csv("fake.csv")

    def test_duplicate_columns_raises(self, converter, mocker):
        """Duplicate column names raise ValueError."""
        mocker.patch.object(
            Path,
            "open",
            return_value=io.StringIO(
                "name,document_id,num_patients,num_patients\nPaper A,1,100,200\n"
            ),
        )
        with pytest.raises(ValueError, match="Duplicate"):
            converter.load_csv("fake.csv")

    def test_invalid_reference_field_key_raises(self, converter, csv_content, mocker):
        """Invalid reference field key raises ValueError."""
        mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
        with pytest.raises(ValueError, match="Invalid mapping keys"):
            converter.load_csv(
                "fake.csv", reference_fields={"not_a_real_field": "num_patients"}
            )

    def test_unknown_attribute_field_raises(self, converter, csv_content, mocker):
        """Unknown attribute field raises ValueError."""
        mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
        with pytest.raises(ValueError, match="Attribute fields not found"):
            converter.load_csv("fake.csv", attribute_fields=["nonexistent_col"])

    def test_headers_normalized_to_lowercase(self, converter, mocker):
        """Headers are stripped and lowercased on load."""
        mocker.patch.object(
            Path,
            "open",
            return_value=io.StringIO("Name,Document_ID,num_patients\nPaper A,1,500\n"),
        )
        colnames, _, _, _ = converter.load_csv("fake.csv")
        assert "name" in colnames
        assert "document_id" in colnames
