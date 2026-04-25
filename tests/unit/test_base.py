"""Tests for core base models."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest
from destiny_sdk.references import ReferenceFileInput

from deet.data_models.base import (
    SUPPORTED_TYPES,
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
    LLMInputSchema,
    coerce_annotation_to_list,
)
from deet.data_models.documents import (
    ContextType,
    Document,
    GoldStandardAnnotatedDocument,
)


def test_attribute_type_to_python_type_population() -> None:
    """Test that the type conversion for Attribute type works."""
    deet_type_str = AttributeType.STRING
    deet_type_int = AttributeType.INTEGER
    deet_type_bool = AttributeType.BOOL
    assert deet_type_str.to_python_type() is str
    assert deet_type_int.to_python_type() is int
    assert deet_type_bool.to_python_type() is bool


@pytest.mark.parametrize("attr_type", list(AttributeType))
def test_to_python_type_is_defined_for_all_enum_members(attr_type):
    """Ensure every AttributeType has a Python type mapping."""
    python_type = attr_type.to_python_type()

    assert isinstance(python_type, type)


@pytest.mark.parametrize(
    ("attr_type", "expected"),
    [
        (AttributeType.BOOL, False),
        (AttributeType.LIST, []),
        (AttributeType.STRING, ""),
        (AttributeType.INTEGER, 0),
        (AttributeType.FLOAT, 0.0),
        (AttributeType.DICT, {}),
    ],
)
def test_missing_annotation_default_matches_type(
    attr_type: AttributeType,
    expected: object,
) -> None:
    """Every current AttributeType defines a missing-annotation placeholder."""
    assert attr_type.missing_annotation_default() == expected


def test_missing_annotation_default_mutable_types_are_fresh() -> None:
    """List and dict defaults must not be shared across calls."""
    first_list = AttributeType.LIST.missing_annotation_default()
    second_list = AttributeType.LIST.missing_annotation_default()
    assert first_list is not second_list
    assert first_list == []

    first_dict = AttributeType.DICT.missing_annotation_default()
    second_dict = AttributeType.DICT.missing_annotation_default()
    assert first_dict is not second_dict
    assert first_dict == {}


def test_attribute_creation_from_dict() -> None:
    """Test creating attribute from dictionary data (as would come from JSON)."""
    # This mimics how attributes are created from JSON data in the annotation converter
    attr_data = {
        "prompt": "Is this a test?",
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": 12345,
        "attribute_label": "Test Boolean Attribute",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.prompt == "Is this a test?"
    assert attr.output_data_type.to_python_type() is bool
    assert attr.attribute_id == 12345
    assert attr.attribute_label == "Test Boolean Attribute"


def test_attribute_creation_with_different_types() -> None:
    """Test creating attributes with different output_data_type values from dict."""
    # Test with str type
    attr_data_str = {
        "prompt": "What is the name?",
        "output_data_type": AttributeType.STRING.value,
        "attribute_id": 12345,
        "attribute_label": "Test String Attribute",
    }
    attr_str = Attribute.model_validate(attr_data_str)
    assert attr_str.output_data_type.to_python_type() is str

    # Test with int type
    attr_data_int = {
        "prompt": "How many items?",
        "output_data_type": AttributeType.INTEGER.value,
        "attribute_id": 123456,
        "attribute_label": "Test Integer Attribute",
    }
    attr_int = Attribute.model_validate(attr_data_int)
    assert attr_int.output_data_type.to_python_type() is int

    # Test with list type
    attr_data_list = {
        "prompt": "What are the items?",
        "output_data_type": AttributeType.LIST.value,
        "attribute_id": 1234567,
        "attribute_label": "Test List Attribute",
    }
    attr_list = Attribute.model_validate(attr_data_list)
    assert attr_list.output_data_type.to_python_type() is list

    # Test with dict type
    attr_data_dict = {
        "prompt": "What are the details?",
        "output_data_type": AttributeType.DICT.value,
        "attribute_id": 123,
        "attribute_label": "Test Dictionary Attribute",
    }
    attr_dict = Attribute.model_validate(attr_data_dict)
    assert attr_dict.output_data_type.to_python_type() is dict

    # Test with float type
    attr_data_float = {
        "prompt": "What is the value?",
        "output_data_type": AttributeType.FLOAT.value,
        "attribute_id": 5432,
        "attribute_label": "Test Float Attribute",
    }
    attr_float = Attribute.model_validate(attr_data_float)
    assert attr_float.output_data_type.to_python_type() is float


def test_attribute_validation_required_fields() -> None:
    """Test that required fields are validated when creating from dict data."""
    # Test that we can create attributes with valid data
    attr_data = {
        "prompt": "Test",
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": 12345,
        "attribute_label": "Test Label",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.prompt == "Test"
    assert attr.attribute_id == 12345
    assert attr.attribute_label == "Test Label"


def test_write_to_csv_creates_new_file(tmp_path) -> None:
    """Test writing attribute to new CSV file."""
    attr = Attribute(
        prompt="Test prompt",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    csv_file = tmp_path / "test.csv"
    attr.write_to_csv(csv_file, mode="w")

    assert csv_file.exists()

    # read back and verify
    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["attribute_id"] == "1234"
        assert rows[0]["prompt"] == "Test prompt"


def test_write_to_csv_appends_to_existing(tmp_path) -> None:
    """Test appending attribute to existing CSV file."""
    attr1 = Attribute(
        prompt="Question 1",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Attribute 1",
    )
    attr2 = Attribute(
        prompt="Question 2",
        output_data_type=AttributeType.STRING,
        attribute_id=2345,
        attribute_label="Attribute 2",
    )

    csv_file = tmp_path / "test.csv"
    attr1.write_to_csv(csv_file, mode="w")
    attr2.write_to_csv(csv_file, mode="a")

    # Read back and verify both rows
    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["attribute_id"] == "1234"
        assert rows[1]["attribute_id"] == "2345"


def test_write_to_csv_creates_parent_directories(tmp_path: Path) -> None:
    """Test that write_to_csv creates parent directories if they don't exist."""
    csv_file = tmp_path / "subdir1" / "subdir2" / "test.csv"

    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.write_to_csv(csv_file)
    assert csv_file.exists()
    assert csv_file.parent.exists()


def test_write_to_csv_overwrites_with_w_mode(tmp_path: Path) -> None:
    """Test that mode='w' overwrites existing file."""
    csv_file = tmp_path / "test.csv"

    attr1 = Attribute(
        prompt="Question 1",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Attribute 1",
    )
    attr2 = Attribute(
        prompt="Question 2",
        output_data_type=AttributeType.STRING,
        attribute_id=2345,
        attribute_label="Attribute 2",
    )

    attr1.write_to_csv(csv_file, mode="w")
    attr2.write_to_csv(csv_file, mode="w")

    # Should only have one row (attr2)
    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["attribute_id"] == "2345"


def test_write_to_csv_with_none_prompt(tmp_path: Path) -> None:
    """Test writing attribute with None prompt value."""
    attr = Attribute(
        prompt=None,
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    csv_file = tmp_path / "test.csv"
    attr.write_to_csv(csv_file)

    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["prompt"] == ""


def test_write_to_csv_includes_all_fields(tmp_path: Path) -> None:
    """Test that all attribute fields are written to CSV."""
    attr = Attribute(
        prompt="Test prompt",
        output_data_type=AttributeType.INTEGER,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    csv_file = tmp_path / "test.csv"
    attr.write_to_csv(csv_file)

    with csv_file.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        row = rows[0]

        assert "prompt" in row
        assert "output_data_type" in row
        assert "attribute_id" in row
        assert "attribute_label" in row


def test_populate_prompt_from_dict_success() -> None:
    """Test successfully populating prompt from dictionary."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "This is a test prompt",
    }

    attr.populate_prompt_from_dict(input_dict)
    assert attr.prompt == "This is a test prompt"


def test_populate_prompt_from_dict_missing_attribute_id() -> None:
    """Test that missing attribute_id raises ValueError."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "prompt": "This is a test prompt",
    }

    with pytest.raises(ValueError, match="input dict must contain"):
        attr.populate_prompt_from_dict(input_dict)


def test_populate_prompt_from_dict_missing_prompt() -> None:
    """Test that missing prompt field raises ValueError."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
    }

    with pytest.raises(ValueError, match="input dict must contain"):
        attr.populate_prompt_from_dict(input_dict)


def test_populate_prompt_from_dict_empty_dict() -> None:
    """Test that empty dictionary raises ValueError."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict: dict = {}

    with pytest.raises(ValueError, match="input dict must contain"):
        attr.populate_prompt_from_dict(input_dict)


def test_populate_prompt_from_dict_mismatched_id() -> None:
    """Test that mismatched attribute_id raises ValueError."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 9999,
        "prompt": "This is a test prompt",
    }

    with pytest.raises(ValueError, match="attribute_id mismatch"):
        attr.populate_prompt_from_dict(input_dict)


def test_populate_prompt_from_dict_string_id() -> None:
    """Test that string attribute_id is converted to int for comparison."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": "1234",
        "prompt": "This is a test prompt",
    }

    attr.populate_prompt_from_dict(input_dict)
    assert attr.prompt == "This is a test prompt"


def test_populate_prompt_overwrites_by_default() -> None:
    """Test that overwrite=True (default) overwrites existing prompt."""
    attr = Attribute(
        prompt="Original prompt",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "New prompt",
    }

    attr.populate_prompt_from_dict(input_dict, overwrite=True)
    assert attr.prompt == "New prompt"


def test_populate_prompt_no_overwrite_with_existing() -> None:
    """Test that overwrite=False preserves existing prompt."""
    attr = Attribute(
        prompt="Original prompt",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "New prompt",
    }

    attr.populate_prompt_from_dict(input_dict, overwrite=False)
    assert attr.prompt == "Original prompt"


def test_populate_prompt_no_overwrite_with_none() -> None:
    """Test that overwrite=False still populates if prompt is None."""
    attr = Attribute(
        prompt=None,
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "New prompt",
    }

    attr.populate_prompt_from_dict(input_dict, overwrite=False)
    assert attr.prompt == "New prompt"


def test_populate_prompt_with_extra_fields() -> None:
    """Test that extra fields in dict are ignored."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "Test prompt",
        "extra_field": "extra value",
        "another_field": 999,
    }

    attr.populate_prompt_from_dict(input_dict)
    assert attr.prompt == "Test prompt"
    # extra fields should be ignored...
    assert not hasattr(attr, "another_field")
    assert not hasattr(attr, "extra_field")


def test_populate_prompt_with_empty_string() -> None:
    """Test populating prompt with empty string."""
    # NOTE: maybe we should change the logic
    # to throw an error rather than let the empty
    # string pass...

    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    input_dict = {
        "attribute_id": 1234,
        "prompt": "",
    }

    attr.populate_prompt_from_dict(input_dict)
    assert attr.prompt == ""


def test_print_tabulated_outputs_table(capsys) -> None:
    """Test that print_tabulated outputs formatted table."""
    attr = Attribute(
        prompt="Test prompt",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.print_tabulated()
    captured = capsys.readouterr()

    # Check that output contains field names and values
    assert "prompt" in captured.out
    assert "Test prompt" in captured.out
    assert "attribute_id" in captured.out
    assert "1234" in captured.out


def test_print_tabulated_with_none_prompt(capsys) -> None:
    """Test print_tabulated with None prompt value."""
    attr = Attribute(
        prompt=None,
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.print_tabulated()
    captured = capsys.readouterr()

    assert "prompt" in captured.out
    # it will simply omit the space where 'None'
    # might be...


def test_print_tabulated_contains_all_fields(capsys) -> None:
    """Test that print_tabulated includes all attribute fields."""
    attr = Attribute(
        prompt="Test prompt",
        output_data_type=AttributeType.STRING,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.print_tabulated()
    captured = capsys.readouterr()

    assert "prompt" in captured.out
    assert "output_data_type" in captured.out
    assert "attribute_id" in captured.out
    assert "attribute_label" in captured.out


@patch("builtins.input", side_effect=["y", "This is my custom prompt", "y"])
def test_enter_custom_prompt_accepts_prompt(mock_input, capsys) -> None:
    """Test entering a custom prompt successfully."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )
    expected_prompt = "This is my custom prompt"
    attr.enter_custom_prompt()

    assert attr.prompt == expected_prompt
    captured = capsys.readouterr()
    assert "Do you want to add a new prompt?" in captured.out
    assert "Confirm? y/n" in captured.out


@patch("builtins.input", side_effect=["y", "This is my custom prompt", "n"])
def test_enter_custom_prompt_user_cancelled(mock_input, capsys) -> None:
    """Test entering a custom prompt successfully."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )
    with pytest.raises(StopIteration):  # input exhausted
        attr.enter_custom_prompt()

    captured = capsys.readouterr()
    assert "Do you want to add a new prompt?" in captured.out
    assert "Confirm? y/n" in captured.out
    assert (
        "Prompt entry cancelled. Please enter again or CTRL+C to exit." in captured.out
    )


@patch("builtins.input", return_value="n")
def test_enter_custom_prompt_declines(mock_input) -> None:
    """Test declining to enter a custom prompt - prompt stays unchanged."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "Test question"


@patch("builtins.input", return_value="N")
def test_enter_custom_prompt_declines_uppercase(mock_input) -> None:
    """Test declining with uppercase N - prompt stays unchanged."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "Test question"


@patch("builtins.input", side_effect=["maybe", "perhaps", "dunno", "n"])
def test_enter_custom_prompt_invalid_then_decline(mock_input, capsys) -> None:
    """Test handling invalid input before declining - prompt stays unchanged."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()

    assert attr.prompt == "Test question"
    captured = capsys.readouterr()
    assert "Please answer either `y` or `n`" in captured.out


@patch("builtins.input", side_effect=["x", "x", "x", "x", "x", "x"])
def test_enter_custom_prompt_max_tries(mock_input) -> None:
    """Test that function returns after max_tries invalid inputs - prompt unchanged."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt(max_tries=5)
    assert attr.prompt == "Test question"


@patch("builtins.input", side_effect=["Y", "This is my custom prompt", "Y"])
def test_enter_custom_prompt_case_insensitive(mock_input) -> None:
    """Test that 'Y' (uppercase) is accepted."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "This is my custom prompt"


@patch("builtins.input", side_effect=["  y  ", "Prompt with whitespace handling", "y"])
def test_enter_custom_prompt_strips_whitespace(mock_input) -> None:
    """Test that whitespace in y/n input is stripped."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "Prompt with whitespace handling"


@patch("builtins.input", side_effect=["y", ""])
def test_enter_custom_prompt_empty_string(mock_input, capsys) -> None:
    """Test entering an empty string as prompt."""
    # NOTE: same as above. may want to
    # raise an error if this occurs instead...
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    with pytest.raises(StopIteration):  # input exhausted
        attr.enter_custom_prompt()
    captured = capsys.readouterr()
    assert "Prompt cannot be empty. Please try again." in captured.out


@patch("builtins.input", side_effect=["invalid", "y", "My prompt", "y"])
def test_enter_custom_prompt_recovers_from_invalid(mock_input) -> None:
    """Test that function recovers from invalid input and accepts valid input."""
    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "My prompt"


def test_document_creation() -> None:
    """Test creating a document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="This is test content",
        context_type=ContextType.FULL_DOCUMENT,
        document_id=1,
        original_doc_filepath=Path("test.pdf"),
    )
    assert doc.name == "Test Document"
    assert doc.document_id == 1
    assert doc.original_doc_filepath == Path("test.pdf")
    assert doc.context == "This is test content"


def test_document_creation_with_list_context() -> None:
    """Test creating a document with context as string (list joined)."""
    citation = ReferenceFileInput()
    context_str = "Paragraph 1\n\nParagraph 2"
    doc = Document(
        name="Test Document 2",
        citation=citation,
        context=context_str,
        context_type=ContextType.RAG_SNIPPETS,
        document_id=2,
    )
    assert doc.context == context_str


def test_gold_standard_annotation_creation_from_dict() -> None:
    """Test creating a gold standard annotation from dictionary data."""
    # This mimics how annotations are created from JSON data
    attr_data = {
        "prompt": "Test question",
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": 1234,
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
        "prompt": "Test question",
        "output_data_type": AttributeType.STRING.value,
        "attribute_id": 2345,
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


def test_gold_standard_annotation_bool_is_booled() -> None:
    """Test that wrong type for bool attribute raises ValueError."""
    attr = Attribute(
        prompt="Is this valid?",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Bool Attribute",
    )

    annotation = GoldStandardAnnotation(
        attribute=attr,
        raw_data="not a bool",
        annotation_type=AnnotationType.HUMAN,
    )

    assert isinstance(annotation.output_data, bool)
    assert annotation.output_data


def test_gold_standard_annotation_string_preserved_on_type_change() -> None:
    """Test that changing attribute type preserves the string value of an annotation."""
    attr = Attribute(
        prompt="Is this valid?",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Bool Attribute",
    )

    string_value = "not a bool"
    annotation = GoldStandardAnnotation(
        attribute=attr,
        raw_data=string_value,
        annotation_type=AnnotationType.HUMAN,
    )
    assert isinstance(annotation.output_data, bool)

    attr.output_data_type = AttributeType.STRING

    assert isinstance(annotation.output_data, str)
    assert annotation.output_data == string_value


@pytest.mark.parametrize(
    ("raw_input", "expected_output"),
    [
        (["A", "B"], ["A", "B"]),
        ([], []),
        ("A;;; B", ["A", "B"]),
        ("A;;; B ;;; C", ["A", "B", "C"]),
        ("1;;; 2.5", ["1", "2.5"]),
        ("A", ["A"]),
        (42, [42]),
        (4.2, [4.2]),
        (True, [True]),
        (None, []),
        (["A", ["B", "C"]], ["A", ["B", "C"]]),
    ],
)
def test_list_coercion(
    raw_input: SUPPORTED_TYPES, expected_output: list[SUPPORTED_TYPES]
):
    result = coerce_annotation_to_list(raw_input)

    assert result == expected_output


def test_gold_standard_annotation_conversion_to_list() -> None:
    """Test that changing attribute type preserves the string value of an annotation."""
    attr = Attribute(
        prompt="Is this valid?",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Bool Attribute",
    )

    annotation = GoldStandardAnnotation(
        attribute=attr,
        raw_data="A;;; B",
        annotation_type=AnnotationType.HUMAN,
    )
    assert isinstance(annotation.output_data, bool)

    attr.output_data_type = AttributeType.LIST

    assert isinstance(annotation.output_data, list)
    assert annotation.output_data == ["A", "B"]


def test_gold_standard_annotated_document_creation() -> None:
    """Test creating a gold standard annotated document."""
    citation = ReferenceFileInput()

    attr = Attribute(
        prompt="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute 3",
    )

    annotation = GoldStandardAnnotation(
        attribute=attr,
        raw_data=True,
        annotation_type=AnnotationType.HUMAN,
    )

    document = Document(
        name="Test Document 3",
        citation=citation,
        context="Test content",
        context_type=ContextType.FULL_DOCUMENT,
        document_id=3,
    )
    doc = GoldStandardAnnotatedDocument(
        document=document,
        annotations=[annotation],
    )
    assert doc.document.name == "Test Document 3"
    assert doc.document.document_id == 3
    assert len(doc.annotations) == 1
    assert doc.annotations[0].output_data is True


def test_llm_input_schema_with_prompt() -> None:
    """Test creating LLMInputSchema when prompt is provided."""
    schema = LLMInputSchema(
        prompt="Custom prompt",
        attribute_id=1234,
        output_data_type=AttributeType.STRING,
    )

    assert schema.prompt == "Custom prompt"


def test_llm_input_schema_fills_from_attribute_label() -> None:
    """Test that fill_prompt fills from attribute_label when prompt is None."""
    data = {
        "prompt": None,
        "attribute_id": 1234,
        "output_data_type": AttributeType.STRING,
        "attribute_label": "Test Attribute Label",
    }

    schema = LLMInputSchema.model_validate(data)
    assert schema.prompt == "Test Attribute Label"


def test_llm_input_schema_preserves_existing_prompt() -> None:
    """Test that fill_prompt doesn't overwrite existing prompt."""
    data = {
        "prompt": "Existing prompt",
        "attribute_id": 1234,
        "output_data_type": AttributeType.STRING,
        "attribute_label": "Test Attribute Label",
    }

    schema = LLMInputSchema.model_validate(data)
    assert schema.prompt == "Existing prompt"


def test_llm_input_schema_ignores_extra_fields() -> None:
    """Test that extra fields are ignored due to Config.extra='ignore'."""
    data = {
        "prompt": "Test prompt",
        "attribute_id": 1234,
        "output_data_type": AttributeType.STRING,
        "extra_field": "should be ignored",
        "another_extra": 999,
    }

    schema = LLMInputSchema.model_validate(data)
    assert schema.prompt == "Test prompt"
    assert not hasattr(schema, "extra_field")
