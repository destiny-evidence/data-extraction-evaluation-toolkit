"""Tests for processed_gold_standard_annotations module."""

import csv
import tempfile
from pathlib import Path

from deet.data_models.base import AttributeType
from deet.data_models.documents import Document, DocumentIdentity
from deet.data_models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
)
from deet.data_models.processed_gold_standard_annotations import (
    ProcessedAnnotationData,
    ProcessedEppiAnnotationData,
)


def test_processed_annotation_data_export_linkage_mapper_csv_basic():
    """Test basic export_linkage_mapper_csv functionality."""
    doc1 = Document(
        name="Test Document 1",
        citation={"title": "Test Title 1", "authors": ["Author 1"]},
        document_id=12345678,
    )
    doc1.document_identity = DocumentIdentity(
        document_id=12345678,
        external_id="EXT123",
        doi=None,
        first_author="Author 1",
        year="2023",
    )

    doc2 = Document(
        name="Test Document 2",
        citation={"title": "Test Title 2", "authors": ["Author 2"]},
        document_id=87654321,
    )
    doc2.document_identity = DocumentIdentity(
        document_id=87654321,
        external_id="EXT456",
        doi=None,
        first_author="Author 2",
        year="2023",
    )

    processed_data = ProcessedAnnotationData(
        attributes=[],
        documents=[doc1, doc2],
        annotations=[],
        annotated_documents=[],
        attribute_id_to_label={},
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        processed_data.export_linkage_mapper_csv(tmp_path)
        with tmp_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        row1 = next(row for row in rows if row["document_id"] == "12345678")
        assert row1["external_id"] == "EXT123"
        assert row1["name"] == "Test Document 1"
        assert row1["file_path"] == ""

        row2 = next(row for row in rows if row["document_id"] == "87654321")
        assert row2["external_id"] == "EXT456"
        assert row2["name"] == "Test Document 2"
        assert row2["file_path"] == ""

    finally:
        tmp_path.unlink(missing_ok=True)


def test_processed_annotation_data_export_linkage_mapper_csv_no_document_identity():
    """Test export_linkage_mapper_csv when document identity is not set."""
    doc1 = Document(
        name="Test Document 1",
        citation={"title": "Test Title 1", "authors": ["Author 1"]},
        document_id=12345678,
    )

    processed_data = ProcessedAnnotationData(
        attributes=[],
        documents=[doc1],
        annotations=[],
        annotated_documents=[],
        attribute_id_to_label={},
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        processed_data.export_linkage_mapper_csv(tmp_path)
        with tmp_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1

        row = rows[0]
        assert row["document_id"] == "12345678"
        assert row["name"] == "Test Document 1"
        assert row["file_path"] == ""

    finally:
        tmp_path.unlink(missing_ok=True)


def test_processed_annotation_data_export_linkage_mapper_csv_no_base_dir():
    """Test export_linkage_mapper_csv when no base directory is provided."""
    doc1 = Document(
        name="Test Document 1",
        citation={"title": "Test Title 1", "authors": ["Author 1"]},
        document_id=12345678,
    )
    doc1.document_identity = DocumentIdentity(
        document_id=12345678,
        external_id="EXT123",
        doi=None,
        first_author="Author 1",
        year="2023",
    )

    doc2 = Document(
        name="Test Document 2",
        citation={"title": "Test Title 2", "authors": ["Author 2"]},
        document_id=87654321,
    )
    doc2.document_identity = DocumentIdentity(
        document_id=87654321,
        external_id="EXT456",
        doi=None,
        first_author="Author 2",
        year="2023",
    )

    processed_data = ProcessedAnnotationData(
        attributes=[],
        documents=[doc1, doc2],
        annotations=[],
        annotated_documents=[],
        attribute_id_to_label={},
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        processed_data.export_linkage_mapper_csv(tmp_path)
        with tmp_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        row1 = next(row for row in rows if row["document_id"] == "12345678")
        assert row1["external_id"] == "EXT123"
        assert row1["name"] == "Test Document 1"
        assert row1["file_path"] == ""

        row2 = next(row for row in rows if row["document_id"] == "87654321")
        assert row2["external_id"] == "EXT456"
        assert row2["name"] == "Test Document 2"
        assert row2["file_path"] == ""

    finally:
        tmp_path.unlink(missing_ok=True)


def test_processed_eppi_annotation_data_inheritance():
    """Test that ProcessedEppiAnnotationData inherits correctly."""
    attribute = EppiAttribute(
        attribute_id=1,
        attribute_label="Test Attribute",
        output_data_type=AttributeType.STRING,
        prompt="Test prompt",
        custom_prompt=None,
    )

    doc = EppiDocument(
        name="Test Document",
        citation={"title": "Test Title", "authors": ["Author"]},
        document_id=12345678,
    )
    doc.document_identity = DocumentIdentity(
        document_id=12345678,
        external_id="EXT123",
        doi=None,
        first_author="Author",
        year="2023",
    )

    annotated_doc = EppiGoldStandardAnnotatedDocument(
        document=doc,
        annotations=[],
    )

    processed_data = ProcessedEppiAnnotationData(
        attributes=[attribute],
        documents=[doc],
        annotations=[],
        annotated_documents=[annotated_doc],
        attribute_id_to_label={1: "Test Attribute"},
        raw_data=None,
    )

    assert processed_data.total_attributes == 1
    assert processed_data.total_documents == 1
    assert processed_data.total_annotations == 0
    assert processed_data.total_annotated_documents == 1

    assert hasattr(processed_data, "export_linkage_mapper_csv")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        processed_data.export_linkage_mapper_csv(tmp_path)
        with tmp_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["document_id"] == "12345678"
        assert row["external_id"] == "EXT123"
        assert row["name"] == "Test Document"

    finally:
        tmp_path.unlink(missing_ok=True)
