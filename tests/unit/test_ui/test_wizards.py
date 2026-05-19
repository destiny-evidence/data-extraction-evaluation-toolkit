from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

from pydantic import BaseModel, Field, SecretStr

from deet.data_models.ui_schema import UI
from deet.ui.terminal.wizards import (
    get_ui_metadata,
    inquire_pydantic_field,
    run_model_wizard,
)


class Color(StrEnum):
    """Sample enum to test wizard generation on enum fields."""

    __test__ = False
    RED = auto()
    BLUE = auto()


class SampleModel(BaseModel):
    """Sample model to test wizard generation."""

    __test__ = False
    str_field: Annotated[str, UI(help="help text for str")] = Field(
        ..., description="Instance str field"
    )
    auto_field: int = Field(default=25)


class ComplexSampleModel(BaseModel):
    """More complicated sample model to test wizard generation of different types."""

    __test__ = False
    str_field: Annotated[str, UI(help="help text for str")] = Field(
        ..., description="Instance str field"
    )
    auto_field: int = Field(default=25)
    path_field: Annotated[Path, UI(help="path hint")] = Field(
        ..., description="path field"
    )
    secret_field: Annotated[SecretStr, UI(help="secret hint")] = Field(
        ..., description="secret field"
    )
    enum_field: Annotated[Color, UI(help="enum hint")] = Field(
        ..., description="enum field"
    )


@patch("deet.ui.terminal.wizards.inquire_pydantic_field")
def test_run_model_wizard_filters_fields(mock_inquire):
    mock_inquire.return_value = "test-str"

    result = run_model_wizard(SampleModel)

    assert mock_inquire.call_count == 1
    assert result.str_field == "test-str"
    assert result.auto_field == 25


def test_get_ui_metadata():
    str_field = SampleModel.model_fields["str_field"]
    auto_field = SampleModel.model_fields["auto_field"]

    str_field_ui_metadata = get_ui_metadata(str_field)
    auto_field_ui_metadata = get_ui_metadata(auto_field)

    assert isinstance(str_field_ui_metadata, UI)
    assert str_field_ui_metadata.help == "help text for str"
    assert auto_field_ui_metadata is None


@patch("InquirerPy.inquirer.filepath")
@patch("InquirerPy.inquirer.secret")
@patch("InquirerPy.inquirer.select")
@patch("InquirerPy.inquirer.text")
def test_widget_mapping(mock_text, mock_select, mock_secret, mock_path):
    for m in [mock_text, mock_select, mock_secret, mock_path]:
        m.return_value.execute.return_value = "dummy_val"

    ui = UI(help="help", valid="valid")

    inquire_pydantic_field(
        ComplexSampleModel,
        "path_field",
        ComplexSampleModel.model_fields["path_field"],
        ui,
    )
    mock_path.assert_called_once()

    inquire_pydantic_field(
        ComplexSampleModel,
        "secret_field",
        ComplexSampleModel.model_fields["secret_field"],
        ui,
    )
    mock_secret.assert_called_once()

    inquire_pydantic_field(
        ComplexSampleModel,
        "enum_field",
        ComplexSampleModel.model_fields["enum_field"],
        ui,
    )
    mock_select.assert_called_once()

    inquire_pydantic_field(
        ComplexSampleModel,
        "str_field",
        ComplexSampleModel.model_fields["str_field"],
        ui,
    )
    mock_text.assert_called_once()


class ValidationModel(BaseModel):
    """Model to test wizard applies validation."""

    __test__ = False
    score: int = Field(..., ge=1, le=10)


@patch("InquirerPy.inquirer.number")
def test_pydantic_validation_logic_in_wizard(mock_number):
    field_name = "score"
    field_info = ValidationModel.model_fields[field_name]
    ui = UI(help="Enter score", valid="Must be 1-10")

    inquire_pydantic_field(ValidationModel, field_name, field_info, ui)

    captured_validator = mock_number.call_args.kwargs["validate"]

    assert captured_validator("5") is True
    assert captured_validator("15") is False
    assert captured_validator("not-a-number") is False
