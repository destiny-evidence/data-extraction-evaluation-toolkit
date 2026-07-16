"""Module containing interactive wizards for collecting information for the deet cli."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Final, cast, get_args

from InquirerPy import inquirer
from InquirerPy.base import BaseSimplePrompt
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from pydantic import BaseModel, SecretStr, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from deet.data_models.ui_schema import UI
from deet.ui.terminal import console
from deet.ui.terminal.components import wizard_field_help, wizard_header

UNCHANGED_SECRET = "<unchanged>"  # noqa: S105


class _GoBack:
    """Sentinel: the user asked to step back to the previous wizard field."""


GO_BACK: Final = _GoBack()

_BACK_KEY: Final = "c-left"
_BACK_FALLBACK_KEY: Final[tuple[str, str]] = ("escape", "b")

_BACK_KEY_BINDINGS: Final[tuple[tuple[str, ...], ...]] = (
    (_BACK_KEY,),
    _BACK_FALLBACK_KEY,
)

PromptResult = str | bool | None | _GoBack


def _execute(prompt: BaseSimplePrompt, *, allow_back: bool) -> PromptResult:
    """
    Run an InquirerPy prompt; return GO_BACK if the back key was pressed.

    When ``allow_back``, back keys are bound eagerly to exit the prompt with GO_BACK
    as its result, so a multi-step wizard can return to the previous field.
    """
    if allow_back:

        def _go_back(event: KeyPressEvent) -> None:
            event.app.exit(result=GO_BACK)

        for keys in _BACK_KEY_BINDINGS:
            prompt.register_kb(*keys, eager=True)(_go_back)

    return prompt.execute()


class WidgetCreator(ABC):
    """Abstract strategy to create Pyinquirer widgets from pydantic fields."""

    @abstractmethod
    def can_handle(self, field_info: FieldInfo) -> bool:
        """Return True if this handler supports the given field."""
        ...

    @abstractmethod
    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Execute the InquirerPy widget and return the validated result."""
        ...


class EnumHandler(WidgetCreator):
    """WidgetCreator to handle enums."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is an enum."""
        return isinstance(field_info.annotation, type) and issubclass(
            field_info.annotation, Enum
        )

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Execute an inquirer.select prompt."""
        enum_type = cast("type[Enum]", field_info.annotation)
        widget_args["choices"] = [e.value for e in enum_type]
        return _execute(inquirer.select(**widget_args), allow_back=allow_back)


class PathHandler(WidgetCreator):
    """WidgetCreator to handle paths."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a Path."""
        return field_info.annotation is Path or Path in get_args(field_info.annotation)

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Execute an inquirer.filepath prompt."""
        return _execute(inquirer.filepath(**widget_args), allow_back=allow_back)


class NumberHandler(WidgetCreator):
    """WidgetCreator to handle numbers."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a number."""
        annotation = field_info.annotation
        return annotation is float or annotation is int or int in get_args(annotation)

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """
        Execute an inquirer.number prompt, adjusted to whether float or not.

        If it's optional, use a text prompt.
        """
        if type(None) in get_args(field_info.annotation):
            answer = _execute(inquirer.text(**widget_args), allow_back=allow_back)
            if answer is GO_BACK:
                return answer
            return str(answer) if answer else None

        if field_info.annotation is float:
            widget_args["float_allowed"] = True
        return _execute(inquirer.number(**widget_args), allow_back=allow_back)


class SecretHandler(WidgetCreator):
    """Widget creator to handle secrets."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if the field is a secretstr."""
        annotation = field_info.annotation
        return annotation is SecretStr or SecretStr in get_args(annotation)

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Execute an inquirer.secret prompt. Leave UNCHANGED_SECRET as None."""
        if field_info.get_default() is None:
            widget_args["default"] = UNCHANGED_SECRET
        answer = _execute(inquirer.secret(**widget_args), allow_back=allow_back)
        return None if answer == UNCHANGED_SECRET else answer


class BoolHandler(WidgetCreator):
    """WidgetCreator to handle boolean fields."""

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Check if field is a boolean."""
        return field_info.annotation is bool or bool in get_args(field_info.annotation)

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Use a inquirer confirm to return boolean."""
        widget_args.pop("validate", None)
        widget_args.pop("filter", None)
        widget_args.pop("invalid_message", None)

        return _execute(inquirer.confirm(**widget_args), allow_back=allow_back)


class DefaultHandler(WidgetCreator):
    """
    Fallback handler (simple text prompt).

    Should always be last in the strategy list.
    """

    def can_handle(self, field_info: FieldInfo) -> bool:
        """Return True, handling whatever is not covered by other strategies."""
        return True

    def execute(
        self, widget_args: dict[str, Any], field_info: FieldInfo, *, allow_back: bool
    ) -> PromptResult:
        """Execute a text prompt."""
        return _execute(inquirer.text(**widget_args), allow_back=allow_back)


STRATEGIES: Final[list[WidgetCreator]] = [
    EnumHandler(),
    PathHandler(),
    NumberHandler(),
    SecretHandler(),
    BoolHandler(),
    DefaultHandler(),
]


def get_ui_metadata(field_info: FieldInfo) -> UI | None:
    """Get UI metadata from pydantic model field."""
    return next((item for item in field_info.metadata if isinstance(item, UI)), None)


def inquire_pydantic_field(  # noqa: PLR0913
    model_class: type[BaseModel],
    field_name: str,
    field_info: FieldInfo,
    ui: UI,
    default_override: str | None = None,
    *,
    allow_back: bool = False,
) -> PromptResult:
    """
    Prompt user to provide data for pydantic field.

    ``default_override`` (a display-ready string: a path, an enum value, or "" for
    an unset optional) replaces the field's own default when supplied, so callers
    can pre-fill the current value while still prompting (e.g. ``deet project edit``).

    ``allow_back`` makes the prompt skippable with Ctrl+Left / Option+Left,
    returning GO_BACK so a multi-step wizard can return to the
    previous field. Standalone prompts leave it False (a single prompt has nowhere
    to go back to).
    """

    def pydantic_validate(answer: str) -> bool | str:
        is_optional = type(None) in get_args(field_info.annotation)
        if is_optional and answer.strip() == "":
            return True
        try:
            model_class.__pydantic_validator__.validate_assignment(
                model_class.model_construct(), field_name, answer
            )
        except ValidationError:
            return False
        else:
            return True

    default = (
        default_override if default_override is not None else field_info.get_default()
    )
    default = "" if default in (PydanticUndefined, None) else default

    widget_args: dict[str, Any] = {
        "message": field_info.description,
        "default": default,
        "validate": lambda ans: pydantic_validate(ans),
        "invalid_message": ui.valid,
        "instruction": ui.instructions,
        "filter": lambda ans: ans.strip() if isinstance(ans, str) else ans,
    }

    for strategy in STRATEGIES:
        if strategy.can_handle(field_info):
            return strategy.execute(widget_args, field_info, allow_back=allow_back)

    not_implemented = f"No widget could be created for field: {field_name}"
    raise NotImplementedError(not_implemented)


def run_model_wizard[T: BaseModel](
    model_class: type[T],
    *,
    prefill: dict[str, object] | None = None,
    defaults: dict[str, str] | None = None,
) -> T:
    """
    Create a wizard from a pydantic model.

    Fields present in ``prefill`` are not prompted for; their values are injected
    into the model directly. This lets a caller supply a field (e.g. the project
    name derived from a directory) instead of asking the user for it.

    ``defaults`` maps a field name to a display-ready string shown as that field's
    editable default, so a caller can pre-fill current values while still prompting
    (e.g. ``deet project edit``).

    Every field after the first is back-navigable: pressing Ctrl+Left or Option+Left
    returns to the previous field, which is re-prompted with the answer already given.
    """
    prefill = prefill or {}
    defaults = defaults or {}
    answers: dict[str, object] = dict(prefill)
    ui_steps: list[tuple[str, FieldInfo, UI]] = [
        (name, info, ui)
        for name, info in model_class.model_fields.items()
        if (ui := get_ui_metadata(info)) is not None and name not in answers
    ]
    total_steps = len(ui_steps)

    index = 0
    while index < len(ui_steps):
        f_name, f_info, f_ui = ui_steps[index]
        console.clear()
        console.print(wizard_header(model_class.__name__, index + 1, total_steps))
        console.print(wizard_field_help(f_name, f_ui.help))
        console.print("Press Ctrl+C to exit", style="dim")
        if index > 0:
            console.print(
                "Press Ctrl+Left/Option+Left to go back",
                style="dim",
            )

        previous_answer = answers.get(f_name)
        seed = (
            previous_answer
            if isinstance(previous_answer, str)
            else defaults.get(f_name)
        )
        result = inquire_pydantic_field(
            model_class,
            f_name,
            f_info,
            f_ui,
            default_override=seed,
            allow_back=index > 0,
        )
        if result is GO_BACK:
            index -= 1
            continue
        answers[f_name] = result
        index += 1

    return model_class.model_validate(answers)


def continue_after_key(message: str = "Press Enter to continue...") -> None:
    """Pause execution until the user acknowledges."""
    inquirer.secret(
        message=message,
        qmark="⌨️ ",
        transformer=lambda _: "",
    ).execute()
