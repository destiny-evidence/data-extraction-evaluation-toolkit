"""Tests for csv_annotation_converter."""

import io
import random
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
def test_bool(converter):
    """Bool strings inferred as bool."""
    assert converter._infer_type("true") is bool
    assert converter._infer_type("False") is bool
    assert converter._infer_type("t") is bool


def test_int(converter):
    """Integer strings inferred as int."""
    assert converter._infer_type("42") is int
    assert converter._infer_type("-3") is int


def test_float(converter):
    """Float strings inferred as float."""
    assert converter._infer_type("3.14") is float


def test_str(converter):
    """Non-numeric strings inferred as str."""
    assert converter._infer_type("hello") is str


def test_empty_returns_none(converter):
    """Empty or whitespace-only strings return None."""
    assert converter._infer_type("") is None
    assert converter._infer_type("   ") is None


# --- _resolve_types ---


def test_all_int(converter):
    """All int types resolve to INTEGER."""
    assert converter._resolve_types([int, None, int]) == AttributeType.INTEGER


def test_int_and_float_resolves_float(converter):
    """Mixed int and float resolves to FLOAT."""
    assert converter._resolve_types([int, float]) == AttributeType.FLOAT


def test_all_str(converter):
    """All str types resolve to STRING."""
    assert converter._resolve_types([str, str]) == AttributeType.STRING


def test_all_none_raises(converter):
    """All-null sample raises ValueError."""
    with pytest.raises(ValueError, match="null"):
        converter._resolve_types([None, None])


def test_incompatible_types_raises(converter):
    """Incompatible types raise ColumnTypeInferenceError."""
    with pytest.raises(ColumnTypeInferenceError):
        converter._resolve_types([str, int])


# --- _find_duplicate_column_names ---


def test_no_duplicates(converter):
    """Returns empty list when no duplicates."""
    assert converter._find_duplicate_column_names(["a", "b", "c"]) == []


def test_finds_duplicate(converter):
    """Returns duplicate names and their positions."""
    result = converter._find_duplicate_column_names(["a", "b", "a"])
    assert result == [{"a": [0, 2]}]


# --- Tests for load_csv ---


@pytest.fixture
def csv_content():
    """Minimal valid CSV string for load_csv tests."""
    return (
        "name,document_id,num_patients,about_adaptation\n"
        "Paper A,1,42,t\n"
        "Paper B,2,70,f\n"
    )


def test_loads_valid_csv(converter, csv_content, mocker):
    """Valid CSV loads with correct columns and row count."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
    colnames, ref_fields, attr_fields, rows = converter.load_csv("fake.csv")
    assert "name" in colnames
    assert "document_id" in colnames
    assert len(rows) == 2


def test_missing_required_column_raises(converter, mocker):
    """Missing required column raises ValueError."""
    mocker.patch.object(
        Path, "open", return_value=io.StringIO("name,num_patients\nPaper A,10\n")
    )
    with pytest.raises(ValueError, match="document_id"):
        converter.load_csv("fake.csv")


def test_duplicate_columns_raises(converter, mocker):
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


def test_invalid_reference_field_key_raises(converter, csv_content, mocker):
    """Invalid reference field key raises ValueError."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
    with pytest.raises(ValueError, match="Invalid mapping keys"):
        converter.load_csv(
            "fake.csv", reference_fields={"not_a_real_field": "num_patients"}
        )


def test_unknown_attribute_field_raises(converter, csv_content, mocker):
    """Unknown attribute field raises ValueError."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(csv_content))
    with pytest.raises(ValueError, match="Attribute fields not found"):
        converter.load_csv("fake.csv", attribute_fields=["nonexistent_col"])


def test_headers_normalized_to_lowercase(converter, mocker):
    """Headers are stripped and lowercased on load."""
    mocker.patch.object(
        Path,
        "open",
        return_value=io.StringIO("Name,Document_ID,num_patients\nPaper A,1,500\n"),
    )
    colnames, _, _, _ = converter.load_csv("fake.csv")
    assert "name" in colnames
    assert "document_id" in colnames


# --- build_attributes ---
def test_infers_integer_attribute(converter):
    """Integer column inferred and attribute built correctly."""
    rows = [{"num_patients": "100"}, {"num_patients": "250"}]
    attrs = converter.build_attributes(["num_patients"], rows)
    assert len(attrs) == 1
    assert attrs[0].output_data_type == AttributeType.INTEGER
    assert attrs[0].attribute_label == "num_patients"


# --- build_destiny_authorship_list ---
def test_single_author(converter):
    """Single author parsed with correct display name."""
    result = converter._build_destiny_authorship_list("Alice")
    assert len(result) == 1
    assert result[0].display_name == "Alice"
    assert result[0].position.value == "first"


def test_multiple_authors_positions(converter):
    """First, middle, and last positions assigned correctly."""
    result = converter._build_destiny_authorship_list("Alice; Bob; Mo")
    assert result[0].position.value == "first"
    assert result[1].position.value == "middle"
    assert result[2].position.value == "last"


def test_empty_string_returns_empty(converter):
    """Empty string returns empty list."""
    assert converter._build_destiny_authorship_list("") == []


# --- Create realistic data ---
def add_white_space(string: str) -> str:
    """Randomly add leading/trailing whitespace to a string."""
    if random.random() < 0.2:  # noqa: S311
        string = " " + string
    if random.random() < 0.2:  # noqa: S311
        string = string + " "
    return string


@pytest.fixture
def sample_csv_fixture() -> str:
    random.seed(42)
    rows = [
        "document_id, name, publication_year, journal, authors, "
        "about_adaptation, num_patients"
    ]

    journal_names = [
        "Nature",
        "PSRM",
        "Science",
        "PNAS",
        "NeurIPS",
        "AAAS",
        "Cell",
        "NEJM",
    ]
    authors = [
        "Smith J.;Johnson L.;Williams K.",
        "Chen Y.;Li X.;Wang Z.",
        "Garcia M.;Rodriguez L.;Martinez R.",
        "Patel R.;Singh A.;Kaur S.",
        "Nguyen T.;Tran P.",
        "Oluwaseun A.;Adebola T.",
        "Kim H.;Park J.;Lee S.",
        "Ahmed S.;Hassan M.",
        "Brown M.;Davis T.",
        "Taylor B.;Anderson C.",
    ]

    # Generate 100 rows with some empty values
    for _ in range(100):
        # Required fields
        document_id = _
        name = f"name_{_}"

        # Bibliographhic info (sometimes empty)
        publication_year = add_white_space(
            random.choice([str(random.randint(1990, 2025)), ""])  # noqa: S311
        )
        journal = add_white_space(random.choice([*journal_names, ""]))  # noqa: S311
        author = add_white_space(random.choice([*authors, ""]))  # noqa: S311

        # Attributes (sometimes empty)
        about_adaptation = add_white_space(random.choice(["t", "f", ""]))  # noqa: S311
        num_patients = add_white_space(
            random.choice([str(random.randint(20, 500)), ""])  # noqa: S311
        )

        rows.append(
            f"{document_id},{name},{publication_year},{journal},{author},{about_adaptation},{num_patients}"
        )

    return "\n".join(rows)


# --- test ProcessAnnotationFile ---
REFERENCE_FIELDS = {
    "publication_year": "publication_year",
    "publication_venue.display_name": "journal",
    "authorship": "authors",
}


def test_produces_expected_counts(converter, sample_csv_fixture, mocker):
    """Processing returns correct number of documents and annotated documents."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(sample_csv_fixture))
    result = converter.process_annotation_file(
        "fake.csv", reference_fields=REFERENCE_FIELDS
    )
    assert len(result.documents) == 100
    assert len(result.annotated_documents) == 100


def test_inferred_attribute_types(converter, sample_csv_fixture, mocker):
    """Attribute types are correctly inferred from column values."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(sample_csv_fixture))
    result = converter.process_annotation_file(
        "fake.csv", reference_fields=REFERENCE_FIELDS
    )
    attr_by_label = {a.attribute_label: a for a in result.attributes}
    assert attr_by_label["about_adaptation"].output_data_type == AttributeType.BOOL
    assert attr_by_label["num_patients"].output_data_type == AttributeType.INTEGER


def test_reference_fields_excluded_from_attributes(
    converter, sample_csv_fixture, mocker
):
    """Reference field columns are not included as attributes."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(sample_csv_fixture))
    result = converter.process_annotation_file(
        "fake.csv", reference_fields=REFERENCE_FIELDS
    )
    attr_labels = {a.attribute_label for a in result.attributes}
    assert "journal" not in attr_labels
    assert "authors" not in attr_labels
    assert "publication_year" not in attr_labels


def test_attribute_id_to_label_mapping(converter, sample_csv_fixture, mocker):
    """Attribute ID to label mapping is consistent with attributes list."""
    mocker.patch.object(Path, "open", return_value=io.StringIO(sample_csv_fixture))
    result = converter.process_annotation_file(
        "fake.csv", reference_fields=REFERENCE_FIELDS
    )
    for attr in result.attributes:
        assert result.attribute_id_to_label[attr.attribute_id] == attr.attribute_label
