"""Module containing interactive wizards for collecting information for the deet cli."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, get_args

from InquirerPy import inquirer
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from rich.align import Align
from rich.panel import Panel

from deet.data_models.ui_schema import UI
from deet.ui import console

T = TypeVar("T", bound=BaseModel)
UNCHANGED_SECRET = "<unchanged>"  # noqa: S105


@dataclass(frozen=True)
class WizardStep:
    """Context for the current step in an interactive wizard."""

    field_name: str
    help_text: str
    current: int
    total: int


def get_ui_metadata(field_info: FieldInfo) -> UI | None:
    """Get UI metadata from pydantic model field."""
    return next((item for item in field_info.metadata if isinstance(item, UI)), None)


def wizard_field_help(field: str, help_text: str) -> None:
    """Print help text to the right."""
    help_card = Panel(
        help_text,
        title=f"About: {field}",
        border_style="cyan",
        padding=(1, 2),
        width=40,
        expand=False,
    )
    help_card = Align.right(help_card)
    console.print(help_card)


def ask_with_help(
    model_class: type[T],
    step: WizardStep,
    prompt_func: Callable,
) -> Callable:
    """Print help text to the right, and then prompt for user input."""
    console.clear()

    header = Panel(
        f"[bold]Setup {model_class.__name__} {step.current}/{step.total}[/]",
        border_style="bright_blue",
        width=30,
    )
    console.print(Align.center(header))

    wizard_field_help(step.field_name, step.help_text)

    return prompt_func()


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
    if default is PydanticUndefined or default is None:
        default = ""

    widget_args: dict[str, Any] = {
        "message": field_info.description,
        "default": default,
        "validate": lambda ans: pydantic_validate(ans),
        "invalid_message": ui.valid,
        "instruction": ui.instructions,
        "filter": lambda ans: ans.strip(),
    }

    if isinstance(field_info.annotation, type) and issubclass(
        field_info.annotation, Enum
    ):
        widget_args["choices"] = [e.value for e in field_info.annotation]
        answer = inquirer.select(**widget_args).execute()
    elif field_info.annotation is Path:
        answer = inquirer.filepath(**widget_args).execute()
    elif field_info.annotation is float:
        widget_args["float_allowed"] = True
        answer = inquirer.number(**widget_args).execute()
    elif field_info.annotation is int or int in get_args(field_info.annotation):
        answer = inquirer.number(**widget_args).execute()
    elif field_info.annotation is SecretStr or SecretStr in get_args(
        field_info.annotation
    ):
        if field_info.get_default() is None:
            widget_args["default"] = UNCHANGED_SECRET
        answer = inquirer.secret(**widget_args).execute()
        if answer == UNCHANGED_SECRET:
            answer = None
    else:
        answer = inquirer.text(**widget_args).execute()

    return answer


def run_model_wizard(model_class: type[T]) -> T:
    """Create a wizard from a pydantic model."""
    answers = {}
    ui_steps: list[tuple[str, FieldInfo, UI]] = []
    for name, info in model_class.model_fields.items():
        ui = get_ui_metadata(info)
        if ui is not None:
            ui_steps.append((name, info, ui))

    total_steps = len(ui_steps)

    for i, (f_name, f_info, f_ui) in enumerate(ui_steps, 1):

        def prompt_wrapper(
            field_name: str = f_name,
            field_info: FieldInfo = f_info,
            ui: UI = f_ui,
        ) -> str | None:
            return inquire_pydantic_field(model_class, field_name, field_info, ui)

        step_context = WizardStep(
            field_name=f_name, help_text=f_ui.help, current=i, total=total_steps
        )

        answer = ask_with_help(
            model_class=model_class,
            step=step_context,
            prompt_func=prompt_wrapper,
        )
        answers[f_name] = answer

    return model_class.model_validate(answers)
