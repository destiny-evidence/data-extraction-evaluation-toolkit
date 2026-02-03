"""Core data models for document processing and annotation."""

import csv
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal

from destiny_sdk.references import ReferenceFileInput
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tabulate import tabulate

MAX_PROMPT_LENGTH = 500
# ruff: noqa: T201, FURB105


class AnnotationType(StrEnum):
    """Enumeration of annotation types."""

    HUMAN = auto()
    LLM = auto()


class AttributeType(StrEnum):
    """Enum of permitted attribute data types."""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()

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


class ContextType(StrEnum):
    """Types of context that can be provided to the LLM."""

    EMPTY = auto()
    FULL_DOCUMENT = auto()
    ABSTRACT_ONLY = auto()
    RAG_SNIPPETS = auto()
    CUSTOM = auto()


class DocumentIDSource(StrEnum):
    """
    Sources for a given document_id. Can be e.g. eppi_item_id.

    To be extended if e.g. we start working with
    non-eppi gold standard references.
    """

    EPPI_ITEM_ID = auto()


class Attribute(BaseModel):
    """
    Core attribute definition for data extraction tasks.

    Represents a single piece of information to be extracted from documents.
    """

    model_config = ConfigDict()

    prompt: str | None = None  # an optional prompt.
    question_target: str  # 'How many patients were recruited?' - the prompt/question
    output_data_type: AttributeType  # One of the defined output data types
    attribute_id: int  # unique identifier for the attribute
    attribute_label: str  # human-readable way of identifying the attribute

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


class Document(BaseModel):
    """
    Represents a document.

    This can be used both for references itemised
    in a document listing gold standard annotations (e.g. eppi.json)
    AND
    for a document coming from a file (e.g. pdf) without
    linking to a gold standard annotations document with references.
    """

    model_config = ConfigDict()

    name: str
    citation: ReferenceFileInput
    context: str | list[str]
    context_type: ContextType
    document_id: int
    document_id_source: DocumentIDSource
    filename: str | None = None


class GoldStandardAnnotation(BaseModel):
    """A single gold standard annotation for an attribute."""

    model_config = ConfigDict()

    attribute: Attribute
    output_data: Any
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
    def ensure_correct_type(cls, data: dict) -> dict:
        """Ensure output_data is of the type required by annotation_type."""
        target_att: Attribute = data["attribute"]

        if isinstance(target_att, dict):
            output_data_type = AttributeType(target_att["output_data_type"])
        else:
            output_data_type = target_att.output_data_type

        target_type: type = output_data_type.to_python_type()

        if not isinstance(data["output_data"], target_type):
            bad_type = (
                f"field {data['output_data']} is of "
                f" type {type(data['output_data'])}; should be {target_type}."
            )
            raise ValueError(bad_type)  # noqa: TRY004 raising ValueError because of pydantic
        return data


class GoldStandardAnnotatedDocument(Document):
    """A document with its gold standard annotations."""

    model_config = ConfigDict()

    annotations: list[GoldStandardAnnotation]


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
    LLM response model for a single annotation.

    This mirrors EppiGoldStandardAnnotation structure but uses attribute_id
    instead of full EppiAttribute object, as the LLM cannot provide the full
    attribute object.
    """

    attribute_id: int = Field(
        ..., description="The ID of the EPPI attribute being annotated"
    )
    output_data: Any = Field(..., description="The LLM's annotation.")
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


class LLMResponseSchema(BaseModel):
    """
    Root schema for LLM annotation extraction response.

    This structure matches the expected format that can be converted
    to list[GoldStandardAnnotation] after attribute resolution.
    """

    annotations: list[LLMAnnotationResponse] = Field(
        ..., description="List of annotations extracted from the document"
    )

    model_config = ConfigDict(extra="forbid")
