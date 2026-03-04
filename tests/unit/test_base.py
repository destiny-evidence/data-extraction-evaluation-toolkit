"""Tests for core base models."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
    LLMInputSchema,
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


def test_attribute_creation_from_dict() -> None:
    """Test creating attribute from dictionary data (as would come from JSON)."""
    # This mimics how attributes are created from JSON data in the annotation converter
    attr_data = {
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": 12345,
        "attribute_label": "Test Boolean Attribute",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.output_data_type.to_python_type() is bool
    assert attr.attribute_id == 12345
    assert attr.attribute_label == "Test Boolean Attribute"


def test_attribute_creation_with_different_types() -> None:
    """Test creating attributes with different output_data_type values from dict."""
    # Test with str type
    attr_data_str = {
        "output_data_type": AttributeType.STRING.value,
        "attribute_id": 12345,
        "attribute_label": "Test String Attribute",
    }
    attr_str = Attribute.model_validate(attr_data_str)
    assert attr_str.output_data_type.to_python_type() is str

    # Test with int type
    attr_data_int = {
        "output_data_type": AttributeType.INTEGER.value,
        "attribute_id": 123456,
        "attribute_label": "Test Integer Attribute",
    }
    attr_int = Attribute.model_validate(attr_data_int)
    assert attr_int.output_data_type.to_python_type() is int

    # Test with list type
    attr_data_list = {
        "output_data_type": AttributeType.LIST.value,
        "attribute_id": 1234567,
        "attribute_label": "Test List Attribute",
    }
    attr_list = Attribute.model_validate(attr_data_list)
    assert attr_list.output_data_type.to_python_type() is list

    # Test with dict type
    attr_data_dict = {
        "output_data_type": AttributeType.DICT.value,
        "attribute_id": 123,
        "attribute_label": "Test Dictionary Attribute",
    }
    attr_dict = Attribute.model_validate(attr_data_dict)
    assert attr_dict.output_data_type.to_python_type() is dict

    # Test with float type
    attr_data_float = {
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
        "output_data_type": AttributeType.BOOL.value,
        "attribute_id": 12345,
        "attribute_label": "Test Label",
    }
    attr = Attribute.model_validate(attr_data)
    assert attr.attribute_id == 12345
    assert attr.attribute_label == "Test Label"


def test_write_to_csv_creates_new_file(tmp_path) -> None:
    """Test writing attribute to new CSV file."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Test prompt",
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Attribute 1",
    )
    attr2 = Attribute(
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Attribute 1",
    )
    attr2 = Attribute(
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt=None,
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
        output_data_type=AttributeType.INTEGER,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Test prompt",
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Original prompt",
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Original prompt",
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt=None,
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Test prompt",
    )

    attr.print_tabulated()
    captured = capsys.readouterr()

    # Check that output contains field names and values
    assert "attribute_id" in captured.out
    assert "1234" in captured.out


def test_print_tabulated_with_none_prompt(capsys) -> None:
    """Test print_tabulated with None prompt value."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt=None,
    )

    attr.print_tabulated()
    captured = capsys.readouterr()

    assert "prompt" in captured.out
    # it will simply omit the space where 'None'
    # might be...


def test_print_tabulated_contains_all_fields(capsys) -> None:
    """Test that print_tabulated includes all attribute fields."""
    attr = Attribute(
        output_data_type=AttributeType.STRING,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Test prompt",
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
    """Test declining to enter a custom prompt."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt is None


@patch("builtins.input", return_value="N")
def test_enter_custom_prompt_declines_uppercase(mock_input) -> None:
    """Test declining with uppercase N."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt is None


@patch("builtins.input", side_effect=["maybe", "perhaps", "dunno", "n"])
def test_enter_custom_prompt_invalid_then_decline(mock_input, capsys) -> None:
    """Test handling invalid input before declining."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()

    assert attr.prompt is None
    captured = capsys.readouterr()
    assert "Please answer either `y` or `n`" in captured.out


@patch("builtins.input", side_effect=["x", "x", "x", "x", "x", "x"])
def test_enter_custom_prompt_max_tries(mock_input) -> None:
    """Test that function returns after max_tries invalid inputs."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt(max_tries=5)
    assert attr.prompt is None


@patch("builtins.input", side_effect=["Y", "This is my custom prompt", "Y"])
def test_enter_custom_prompt_case_insensitive(mock_input) -> None:
    """Test that 'Y' (uppercase) is accepted."""
    attr = Attribute(
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
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
    )

    attr.enter_custom_prompt()
    assert attr.prompt == "My prompt"


def test_gold_standard_annotation_creation_from_dict() -> None:
    """Test creating a gold standard annotation from dictionary data."""
    # This mimics how annotations are created from JSON data
    attr_data = {
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


def test_gold_standard_annotation_bool_type_invalid() -> None:
    """Test that wrong type for bool attribute raises ValueError."""
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Bool Attribute",
    )

    with pytest.raises(ValueError, match="should be"):
        GoldStandardAnnotation(
            attribute=attr,
            output_data="not a bool",
            annotation_type=AnnotationType.HUMAN,
        )


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
