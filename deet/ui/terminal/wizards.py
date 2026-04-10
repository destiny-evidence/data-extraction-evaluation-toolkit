"""Module containing interactive wizards for collecting information for the deet cli."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast, get_args

from InquirerPy import inquirer
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from deet.data_models.ui_schema import UI
from deet.ui.terminal import console
from deet.ui.terminal.components import wizard_field_help, wizard_header

UNCHANGED_SECRET = "<unchanged>"  # noqa: S105


class WidgetCreator(ABC):
    """Abstract strategy to create Pyinquirer widgets from pydantic fields."""

    @abstractmethod
    def can_handle(self, field_info: FieldInfo) -> bool:
        """Return True if this handler supports the given field."""
        ...

    @abstractmethod
    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str | None:
        """Execute the InquirerPy widget and return the validated result."""
        ...


class EnumHandler(WidgetCreator):
    """WidgetCreator to handle enums."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is an enum."""
        return isinstance(field_info.annotation, type) and issubclass(
            field_info.annotation, Enum
        )

    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str:
        """Execute an inquirer.select prompt."""
        enum_type = cast(type[Enum], field_info.annotation)
        widget_args["choices"] = [e.value for e in enum_type]
        return inquirer.select(**widget_args).execute()


class PathHandler(WidgetCreator):
    """WidgetCreator to handle paths."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a Path."""
        return field_info.annotation is Path

    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str:
        """Execute an inquirer.filepath prompt."""
        return inquirer.filepath(**widget_args).execute()


class NumberHandler(WidgetCreator):
    """WidgetCreator to handle numbers."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a number."""
        annotation = field_info.annotation
        return annotation is float or annotation is int or int in get_args(annotation)

    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str:
        """Execute an inquirer.number prompt, adjusted to whether float or not."""
        if field_info.annotation is float:
            widget_args["float_allowed"] = True
        return inquirer.number(**widget_args).execute()


class SecretHandler(WidgetCreator):
    """Widget creator to handle secrets."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a secretstr."""
        annotation = field_info.annotation
        return annotation is SecretStr or SecretStr in get_args(annotation)

    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str | None:
        """Execute an inquirer.secret prompt. Leave UNCHANGED_SECRET as None."""
        if field_info.get_default() is None:
            widget_args["default"] = UNCHANGED_SECRET
        answer: str = inquirer.secret(**widget_args).execute()
        return None if answer == UNCHANGED_SECRET else answer


class DefaultHandler(WidgetCreator):
    """Fallback handler. Should always be last in the strategy list."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Return True, handling whatever is not covered by other strategies."""
        return True

    def execute(self, widget_args: dict[str, Any], field_info: FieldInfo) -> str:
        """Execute a text prompt."""
        return inquirer.text(**widget_args).execute()


STRATEGIES: Final[list[WidgetCreator]] = [
    EnumHandler(),
    PathHandler(),
    NumberHandler(),
    SecretHandler(),
    DefaultHandler(),
]


def get_ui_metadata(field_info: FieldInfo) -> UI | None:
    """Get UI metadata from pydantic model field."""
    return next((item for item in field_info.metadata if isinstance(item, UI)), None)


def inquire_pydantic_field(
    model_class: type[BaseModel], field_name: str, field_info: FieldInfo, ui: UI
) -> str | None:
    """Prompt user to provide data for pydantic field."""

    def pydantic_validate(answer: str) -> bool | str:
        try:
            model_class.__pydantic_validator__.validate_assignment(
                model_class.model_construct(), field_name, answer
            )
        except ValidationError:
            return False
        else:
            return True

    default = field_info.get_default()
    default = "" if default in (PydanticUndefined, None) else default

    widget_args: dict[str, Any] = {
        "message": field_info.description,
        "default": default,
        "validate": lambda ans: pydantic_validate(ans),
        "invalid_message": ui.valid,
        "instruction": ui.instructions,
        "filter": lambda ans: ans.strip(),
    }

    for strategy in STRATEGIES:
        if strategy.can_handle(field_info):
            return strategy.execute(widget_args, field_info)

    not_implemented = f"No widget could be created for field: {field_name}"
    raise NotImplementedError(not_implemented)


def run_model_wizard[T: BaseModel](model_class: type[T]) -> T:
    """Create a wizard from a pydantic model."""
    answers = {}
    ui_steps: list[tuple[str, FieldInfo, UI]] = []
    for name, info in model_class.model_fields.items():
        ui = get_ui_metadata(info)
        if ui is not None:
            ui_steps.append((name, info, ui))

    total_steps = len(ui_steps)

    for i, (f_name, f_info, f_ui) in enumerate(ui_steps, 1):
        console.clear()
        console.print(wizard_header(model_class.__name__, i, total_steps))
        console.print(wizard_field_help(f_name, f_ui.help))

        answers[f_name] = inquire_pydantic_field(model_class, f_name, f_info, f_ui)

    return model_class.model_validate(answers)


def continue_after_key(message: str = "Press Enter to continue...") -> None:
    """Pause execution until the user acknowledges."""
    inquirer.secret(
        message=message,
        qmark="⌨️ ",
        transformer=lambda _: "",
    ).execute()
