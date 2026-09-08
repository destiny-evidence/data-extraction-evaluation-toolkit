"""Core data models regarding annotations."""

import csv
from collections.abc import Callable, Iterator
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, Never, TypeVar, cast

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    create_model,
    model_validator,
)
from pydantic.config import JsonDict, JsonValue
from tabulate import tabulate

MAX_PROMPT_LENGTH = 500
# ruff: noqa: T201, FURB105


class AnnotationType(StrEnum):
    """Enumeration of annotation types."""

    HUMAN = auto()
    LLM = auto()
    KEYWORD = auto()


class AttributeType(StrEnum):
    """Enum of permitted attribute data types."""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()

    def missing_annotation_default(
        self,
    ) -> bool | str | int | float | list[Never] | dict[str, Never]:
        """
        Return default ``output_data`` when no gold-standard annotation exists.

        Used when synthesizing a placeholder annotation (e.g. comparing LLM output
        to gold standard where a value was never annotated).

        Returns a fresh ``list`` or ``dict`` for mutable types so callers do not share
        state.

        Raises:
            ValueError: If this member has no defined default.

        Note:
            This is not ``Enum._missing_``; that hook resolves *unrecognised raw
            values* when constructing enum members, not per-type defaults.

        """
        match self:
            case AttributeType.BOOL:
                return False
            case AttributeType.LIST:
                return []
            case AttributeType.STRING:
                return ""
            case AttributeType.INTEGER:
                return 0
            case AttributeType.FLOAT:
                return 0.0
            case AttributeType.DICT:
                return {}
            case _:
                unsupported = (
                    f"No default for missing annotation when attribute type is {self!s}"
                )
                raise ValueError(unsupported)

    def __str__(self) -> str:
        """Return the string value for JSON serialization."""
        return self.value

    def to_python_type(self) -> type:
        """Map AttributeType to actual Python types."""
        mapping = {
            AttributeType.STRING: str,
            AttributeType.INTEGER: int,
            AttributeType.FLOAT: float,
            AttributeType.BOOL: bool,
            AttributeType.LIST: list,
            AttributeType.DICT: dict,
        }
        return mapping[self]

    def llm_annotation_response_model(self) -> type[BaseModel]:
        """
        Return the shared Pydantic sub-model for LLM responses of this type.

        One model is built and cached per :class:`AttributeType`; attributes
        with the same type reuse it in :func:`build_llm_response_model`.

        Returns
            A Pydantic model class with typed ``output_data`` for this type.

        """
        return _llm_annotation_response_model_for_type(self)

    def to_json_type(self) -> JsonValue:
        """Map AttributeType to JS types for the JSON schema."""
        mapping: JsonDict = {
            AttributeType.STRING: {"type": "string"},
            AttributeType.INTEGER: {"type": "integer"},
            AttributeType.FLOAT: {"type": "number"},
            AttributeType.BOOL: {"type": "boolean"},
            AttributeType.LIST: {
                "type": "array",
                "items": {"type": "object", "additionalProperties": False},
            },
            AttributeType.DICT: {"type": "object", "additionalProperties": False},
        }
        return mapping[self]


DEFAULT_ATTRIBUTE_TYPE = AttributeType.BOOL
SUPPORTED_TYPES = str | int | float | bool | list | dict


class Attribute(BaseModel):
    """
    Core attribute definition for data extraction tasks.

    Represents a single piece of information to be extracted from documents.
    """

    model_config = ConfigDict()

    prompt: str | None = None  # an optional prompt.
    output_data_type: AttributeType  # One of the defined output data types
    attribute_id: int  # unique identifier for the attribute
    attribute_label: str  # human-readable way of identifying the attribute

    concept_id: str | None = Field(
        default=None,
        description="ID of the concept this atttribute maps to in a vocabulary",
    )

    def write_to_csv(self, filepath: Path, mode: Literal["a", "w"] = "a") -> None:
        """
        Write an attribute as a line to a csv file - fields represent columns.

        Args:
            filepath (Path): outfile destination.
            mode (Literal["a", "w"], optional): _w_rite or _a_ppend.
            Defaults to "a" (append).

        """
        dictified = self.model_dump()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        write_header = not file_exists or mode == "w"

        with filepath.open(mode=mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dictified.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(dictified)

        logger.debug(f"Wrote attribute to {filepath}")

    def populate_prompt_from_dict(
        self, input_dict: dict[str, Any], *, overwrite: bool = True
    ) -> None:
        """
        Populate the `prompt` field in an Attribute instance from a dict.

        The dict must contain following fields:
            - attribute_id
            - prompt
        and attribute_id(dict) must match self.attribute_id.

        NOTE: this would typically be used in a loop to populate
        prompts for a list of attributes from a csv file where every
        row represents an attribute.

        Args:
            input_dict (dict[str, Any]): An input dict, typically a line in a csv file.
            overwrite (bool, optional): Overwrite existing val in `self.prompt`.
            Defaults to True.

        """
        for field in ["attribute_id", "prompt"]:
            if field not in input_dict:
                bad_dict = (
                    "input dict must contain at least `attribute_id` and `prompt`"
                    " fields. currently, it only "
                    f"contains: {', '.join(input_dict.keys())}"
                )
                raise ValueError(bad_dict)

        if int(input_dict["attribute_id"]) != self.attribute_id:
            bad_att_id = (
                f"attribute_id mismatch: input: {input_dict['attribute_id']}. "
                f" self: {self.attribute_id}"
            )
            raise ValueError(bad_att_id)

        if overwrite or (not overwrite and self.prompt is None):
            self.prompt = input_dict["prompt"]
            logger.debug("added prompt  [...] to Attribute instance.")
        else:
            logger.info("overwrite is set to False, no overwrite prompts.")

    def print_tabulated(self) -> None:
        """Print tabulated version of the contents of this attribute."""
        dictified = self.model_dump()
        data = [[k, v] for k, v in dictified.items()]

        print(tabulate(data, headers=["Field", "Value"], tablefmt="simple"))

    def enter_custom_prompt(self, max_tries: int = 5) -> None:
        """Use CLI to add a prompt."""
        self.print_tabulated()
        print("")
        print("Do you want to add a new prompt? y/n. Use CTRL+C to cancel.")
        tries = 0
        while True:
            user_input = input().strip().lower()

            if user_input == "n":
                logger.debug("user chose not to write a prompt...")
                return

            if user_input == "y":
                break

            print("Please answer either `y` or `n`...")
            tries += 1
            if tries >= max_tries:
                return

        def sanitize_prompt(prompt: str) -> str:
            # Remove non-printable/control characters
            return "".join(c for c in prompt if c.isprintable())

        while True:
            print(f"Please enter your prompt (max {MAX_PROMPT_LENGTH} characters): ")
            user_prompt = input().strip()
            user_prompt = sanitize_prompt(user_prompt)
            if len(user_prompt) == 0:
                print("Prompt cannot be empty. Please try again.")
                continue
            if len(user_prompt) > MAX_PROMPT_LENGTH:
                print(f"Prompt exceeds max {MAX_PROMPT_LENGTH} chars. Shorten!.")
                continue
            print("\nYour prompt will be stored as:\n")
            print(f'"{user_prompt}"')
            print("Confirm? y/n")
            confirm = input().strip().lower()
            if confirm == "y":
                self.prompt = user_prompt
                logger.debug(f"wrote prompt {self.prompt[:30]} [...] to prompt field.")
                return
            if confirm == "n":
                print("Prompt entry cancelled. Please enter again or CTRL+C to exit.")
                continue


AttributeTypeVar = TypeVar("AttributeTypeVar", bound=Attribute)


def coerce_annotation_to_str(val: SUPPORTED_TYPES) -> str:
    """Coerce an annotation to a string."""
    return str(val) if val else ""


def coerce_annotation_to_bool(val: SUPPORTED_TYPES) -> bool:
    """Coerce an annotation to a bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str) and val.lower() in ("false", "0"):
        return False

    if isinstance(val, int | float):
        return bool(val)

    return True


def coerce_annotation_to_int(val: SUPPORTED_TYPES) -> int | None:
    """Coerce an annotation to a int."""
    if isinstance(val, int):
        return val
    if isinstance(val, str | float | bool):
        try:
            return int(val)
        except ValueError:
            logger.warning("Could not convert {val} to int")

    logger.warning(f"Unsupported type for int conversion: {type(val).__name__}")
    return None


def coerce_annotation_to_float(val: SUPPORTED_TYPES) -> float | None:
    """Coerce an annotation to a float."""
    if isinstance(val, float):
        return val
    if isinstance(val, str | bool | int):
        try:
            return float(val)
        except ValueError:
            logger.warning(f"Could not convert {val} to float")

    logger.warning(f"Unsupported type for float conversion: {type(val).__name__}")
    return None


def coerce_annotation_to_list(val: SUPPORTED_TYPES) -> list:
    """Coerce an annotation to list."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        list_of_strings = [item.strip() for item in val.split(";;;")]
        try:
            return [int(item) for item in list_of_strings]
        except (ValueError, TypeError):
            logger.debug("Could not convert items to int.")

        try:
            return [float(item) for item in list_of_strings]
        except (ValueError, TypeError):
            logger.debug("Could not convert items to float")

        return list_of_strings
    return [val]


ANNOTATION_COERCION_STRATEGIES: dict[
    AttributeType, Callable[[SUPPORTED_TYPES], SUPPORTED_TYPES | None]
] = {
    AttributeType.STRING: coerce_annotation_to_str,
    AttributeType.BOOL: coerce_annotation_to_bool,
    AttributeType.INTEGER: coerce_annotation_to_int,
    AttributeType.FLOAT: coerce_annotation_to_float,
    AttributeType.LIST: coerce_annotation_to_list,
}


class GoldStandardAnnotation(BaseModel):
    """
    A single gold standard annotation for an attribute.

    `raw_data` stores the data as it comes from source,
    `output_data` is computed and coerces raw_data into the correct type.
    This can change if the `AttributeType` of the `attribute` changes.
    """

    attribute: Attribute
    raw_data: Any = Field(
        description=(
            "The output data exactly as it was first seen"
            " without any coercion to the correct type"
        )
    )
    annotation_type: AnnotationType
    additional_text: str | None = Field(
        description="Notes provided by the annotator - usually the citation "
        " from the paper containing the context window where the attribute is found",
        default=None,
    )
    reasoning: str | None = Field(
        description="Reasoning, taken from LLM response", default=None
    )

    @model_validator(mode="before")
    @classmethod
    def handle_output_data_input(cls, data: SUPPORTED_TYPES) -> SUPPORTED_TYPES:
        """Catch instantations with output_data and send this to raw_data."""
        if isinstance(data, dict) and "output_data" in data and "raw_data" not in data:
            data["raw_data"] = data.pop("output_data")
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def output_data(self) -> SUPPORTED_TYPES | None:
        """Coerce raw data to correct type based on attribute."""
        strategy = ANNOTATION_COERCION_STRATEGIES.get(self.attribute.output_data_type)

        if strategy:
            return strategy(self.raw_data)

        return self.raw_data


GoldStandardAnnotationTypeVar = TypeVar(
    "GoldStandardAnnotationTypeVar", bound=GoldStandardAnnotation
)


# models specifically for interfacing with the LLM below
class LLMInputSchema(BaseModel):
    """Schema for data going into the LLM."""

    prompt: str
    attribute_id: int
    output_data_type: AttributeType

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def fill_prompt(cls, data: dict, fill_from_field: str = "attribute_label") -> dict:
        """
        Fill `prompt` field if empty.

        Args:
            data (dict): the incoming data
            fill_from_field (str, optional): field to use to fill prompt if empty.
                Defaults to "attribute_label".

        Returns:
            dict: the populated data.

        """
        if data["prompt"] is not None:
            return data
        logger.debug(data)
        if fill_from_field not in data:
            no_fill_field = f" '{fill_from_field}' is missing from data"
            raise ValueError(no_fill_field)
        data["prompt"] = data[fill_from_field]
        logger.debug(f"filled `prompt` with {data['prompt']}.")
        return data


class LLMAnnotationResponse(BaseModel):
    """
    LLM response model for a single attribute's annotation.

    Used as the base for the per-type sub-models produced by
    :meth:`AttributeType.llm_annotation_response_model`. The attribute identity
    is *not* stored on this model; it is encoded in the parent field name
    (``attribute_<id>``) so the LLM is never asked to repeat (and potentially
    mismatch) the id.

    ``output_data`` is typed ``Any`` here and is always overridden with the
    attribute's concrete Python type when the per-type sub-model is built.
    """

    output_data: Any = Field(
        ...,
        description="The LLM's annotation for this attribute.",
    )
    additional_text: str | None = Field(
        ...,
        description=(
            "Supporting text from document containing the context window "
            "where the attribute is found"
        ),
    )
    reasoning: str | None = Field(
        ...,
        description="Reasoning or explanation for the annotation decision",
    )

    # Note: arm_id, arm_title, arm_description, item_attribute_full_text_details
    # are not included as they're EPPI-specific metadata the LLM cannot provide

    model_config = ConfigDict(extra="forbid")


class StaticLLMAnnotationResponse(LLMAnnotationResponse):
    """
    Untyped LLM annotation response model where attribute_id is defined per
    annotation.
    """

    attribute_id: int = Field(
        ..., description="The ID of the attribute being annotated."
    )


class BaseLLMResponse(BaseModel):
    """Base for all LLM response models."""

    model_config = ConfigDict(extra="forbid")


class LLMResponseSchema(BaseLLMResponse):
    """
    Static response schema containing a list of StaticLLMAnnotationResponses.

    This structure contains a list of StaticLLMAnnotationResponses, where
    each response has an attribute_id, and output_data_type is untyped.

    Responses of this type are cheaper to request, since the json schema passed to
    the llm is shorter. However such a schema does not require llms to produce
    exactly one annotation per attribute, or to make sure that output_data_type
    matches that defined at the attribute level.
    """

    annotations: list[StaticLLMAnnotationResponse] = Field(
        ..., description="List of annotations extracted from the document"
    )


class DynamicLLMResponseBase(BaseLLMResponse):
    """
    The base for dynamically generated schemas.

    We can expect that each field is typed as a subclass of LLMAnnotationResponse.
    """

    def iter_attribute_responses(self) -> Iterator[tuple[str, LLMAnnotationResponse]]:
        """Yield field names and safely typed LLMAnnotationResponses."""
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, LLMAnnotationResponse):
                yield field_name, value


_llm_annotation_response_models: dict[AttributeType, type[BaseModel]] = {}


def _llm_annotation_response_model_for_type(
    attribute_type: AttributeType,
) -> type[BaseModel]:
    """
    Build or return the cached LLM annotation sub-model for an attribute type.

    Args:
        attribute_type: The attribute data type to model.

    Returns:
        A Pydantic model class with ``output_data`` typed for ``attribute_type``.

    """
    cached = _llm_annotation_response_models.get(attribute_type)
    if cached is not None:
        return cached

    output_type = attribute_type.to_python_type()
    model = create_model(
        f"LLM{attribute_type.name}AnnotationResponse",
        __base__=LLMAnnotationResponse,
        output_data=(
            output_type,
            Field(..., description="The LLM's annotation for this attribute."),
        ),
    )
    _llm_annotation_response_models[attribute_type] = model
    return model


ATTRIBUTE_RESPONSE_KEY_PREFIX = "attribute_"


def attribute_response_key(attribute_id: int) -> str:
    """
    Build the dynamic-model field name used for a given attribute.

    Args:
        attribute_id: The attribute's unique identifier.

    Returns:
        The field name, e.g. ``"attribute_1234"``.

    """
    return f"{ATTRIBUTE_RESPONSE_KEY_PREFIX}{attribute_id}"


def attribute_id_from_response_key(key: str) -> int:
    """
    Recover the attribute id from a dynamic-model field name.

    Args:
        key: A field name such as ``"attribute_1234"``.

    Returns:
        The integer attribute id encoded in the key.

    Raises:
        ValueError: If ``key`` does not encode a valid integer attribute id.

    """
    raw_id = key.removeprefix(ATTRIBUTE_RESPONSE_KEY_PREFIX)
    try:
        return int(raw_id)
    except ValueError as exc:
        msg = f"Cannot recover attribute id from response key {key!r}"
        raise ValueError(msg) from exc


def build_llm_response_model(
    attributes: list[Attribute],
) -> type[DynamicLLMResponseBase]:
    """
    Build a dynamic LLM response model from the selected attributes.

    Each attribute becomes a required, correctly typed sub-model keyed by
    ``attribute_<id>`` on the returned root model. Because every key is required
    and ``extra="forbid"`` is set at every level, the JSON schema forces the LLM
    to return exactly one response per attribute - no missing attributes, no
    extra/hallucinated attributes, and ``output_data`` constrained to the
    attribute's concrete type. This also yields a schema that providers such as
    Ollama accept as valid (unlike an ``Any``-typed field).

    Args:
        attributes: The attributes to extract; must be non-empty.

    Returns:
        A dynamically created Pydantic model class suitable for use as a
        structured-output schema and for validating the LLM response.

    Raises:
        ValueError: If ``attributes`` is empty.

    """
    if not attributes:
        msg = "Cannot build an LLM response model from an empty attribute list"
        raise ValueError(msg)

    response_fields: dict[str, tuple[type[BaseModel], Any]] = {}
    for attribute in attributes:
        response_fields[attribute_response_key(attribute.attribute_id)] = (
            attribute.output_data_type.llm_annotation_response_model(),
            Field(...),
        )

    return create_model(
        "DynamicLLMResponse",
        __base__=DynamicLLMResponseBase,
        **cast(Any, response_fields),
    )
