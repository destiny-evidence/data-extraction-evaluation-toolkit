"""Core data models for document processing and annotation."""

import csv
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

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


AttributeTypeVar = TypeVar("AttributeTypeVar", bound=Attribute)


class Document(BaseModel):
    """
    Represents a document.

    This can be used both for references itemised
    in a document listing gold standard annotations (e.g. eppi.json)
    AND
    for a document coming from a file (e.g. pdf) without
    linking to a gold standard annotations document with references.
    """

    name: str
    citation: ReferenceFileInput
    context: str | list[str]
    context_type: ContextType
    document_id: int
    document_id_source: DocumentIDSource
    filename: str | None = None


DocumentTypeVar = TypeVar("DocumentTypeVar", bound=Document)


class GoldStandardAnnotation(BaseModel):
    """A single gold standard annotation for an attribute."""

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


GoldStandardAnnotationTypeVar = TypeVar(
    "GoldStandardAnnotationTypeVar", bound=GoldStandardAnnotation
)


class GoldStandardAnnotatedDocument(Document, Generic[GoldStandardAnnotationTypeVar]):
    """A document with its gold standard annotations."""

    annotations: list[GoldStandardAnnotationTypeVar]


GoldStandardAnnotatedDocumentTypeVar = TypeVar(
    "GoldStandardAnnotatedDocumentTypeVar", bound=GoldStandardAnnotatedDocument
)


class ProcessedAttributeData(BaseModel, Generic[AttributeTypeVar]):
    """
    Structured result from annotation processing.

    Contains only attributes, so the ProcessedAnnotationData class can
    subclass this
    """

    attributes: list[AttributeTypeVar]

    def _custom_prompts_cli(self) -> None:
        """
        Use an interactive CLI to have the user enter custom prompts.

        Args:
            attribute (Attribute): a single (Eppi)Attribute

        """
        for attribute in self.attributes:
            attribute.enter_custom_prompt()

    def export_attributes_csv_file(self, filepath: Path) -> None:
        """
        Write a csv file containing all attributes for prompt population.

        Args:
            filepath (Path): outfile path.


        """
        if filepath.suffix != ".csv":
            bad_filetype = "file ending must be .csv"
            raise ValueError(bad_filetype)
        for attribute in self.attributes:
            attribute.write_to_csv(filepath=filepath)

        logger.info(f"wrote attributes to file {filepath}.")

    def _import_prompts_csv_file(
        self, filepath: Path, *, overwrite: bool = True
    ) -> None:
        """
        Import prompts from a csv file.

        Args:
            filepath (Path): attribute/prompt input file.
            overwrite (bool, optional): Overwrite existing prompts. Defaults to True.

        """
        if not filepath.exists():
            no_file = f"CSV file not found: {filepath}"
            raise FileNotFoundError(no_file)

        if filepath.suffix != ".csv":
            bad_suffix = "File must have .csv extension"
            raise ValueError(bad_suffix)

        with filepath.open(mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                empty_csv = "CSV file is empty or has no headers"
                raise ValueError(empty_csv)

            required_fields = ["attribute_id", "prompt"]
            for field in required_fields:
                if field not in reader.fieldnames:
                    csv_missing_fields = (
                        f"CSV must contain '{field}' column. "
                        f"Found columns: {', '.join(reader.fieldnames)}"
                    )

                    raise ValueError(csv_missing_fields)

            rows_processed = 0
            for row in reader:
                # find attribute_id match
                attribute_id = int(row["attribute_id"])
                matching_attribute = None

                for attribute in self.attributes:
                    if attribute.attribute_id == attribute_id:
                        matching_attribute = attribute
                        break

                if matching_attribute is None:
                    logger.warning(
                        f"No attribute found with ID {attribute_id}, skipping row"
                    )
                    continue

                # populate prompt using the Attribute method
                try:
                    matching_attribute.populate_prompt_from_dict(
                        row, overwrite=overwrite
                    )
                    rows_processed += 1
                except ValueError as e:
                    logger.error(
                        f"Error processing row for attribute {attribute_id}: {e}"
                    )

            logger.info(f"Processed {rows_processed} prompts from {filepath}")

    def populate_custom_prompts(
        self, method: Literal["cli", "file"], filepath: Path | None = None, **kwargs
    ) -> None:
        """
        Populate custom prompts.

        Args:
            method (Literal["cli", "file"])
            filepath (Path | None): infile path.

        Raises:
            FileNotFoundError: if method is file and there's no filepath.

        """
        if method == "cli":
            self._custom_prompts_cli()
        elif method == "file":
            if filepath is None:
                missing_filepath = "please specify a filepath!"
                raise FileNotFoundError(missing_filepath)
            self._import_prompts_csv_file(filepath=filepath, **kwargs)
        else:
            not_impl = f"method {method} is not implemented. use cli or file."
            raise NotImplementedError(not_impl)

    @property
    def total_attributes(self) -> int:
        """Total number of attributes processed."""
        return len(self.attributes)


class ProcessedAnnotationData(
    ProcessedAttributeData,
    Generic[
        AttributeTypeVar,
        DocumentTypeVar,
        GoldStandardAnnotationTypeVar,
        GoldStandardAnnotatedDocumentTypeVar,
    ],
):
    """
    Structured result from annotation processing.

    This model provides a clean, validated structure for all processed
    annotation data with useful properties and methods.
    """

    documents: list[DocumentTypeVar]
    annotations: list[GoldStandardAnnotationTypeVar]
    annotated_documents: list[GoldStandardAnnotatedDocumentTypeVar]
    attribute_id_to_label: dict[int, str]

    @property
    def total_documents(self) -> int:
        """Total number of documents processed."""
        return len(self.documents)

    @property
    def total_annotations(self) -> int:
        """Total number of annotations processed."""
        return len(self.annotations)

    @property
    def total_annotated_documents(self) -> int:
        """Total number of documents with annotations."""
        return len(self.annotated_documents)

    def get_documents_with_annotations(self) -> list[Document]:
        """Get only documents that have annotations."""
        annotated_doc_ids = {doc.document_id for doc in self.annotated_documents}
        return [doc for doc in self.documents if doc.document_id in annotated_doc_ids]

    def get_annotations_by_type(
        self, annotation_type: AnnotationType
    ) -> list[GoldStandardAnnotation]:
        """Get all annotations of a specific type (human/llm)."""
        return [
            ann for ann in self.annotations if ann.annotation_type == annotation_type
        ]

    def get_attribute_by_id(self, attribute_id: int) -> Attribute | None:
        """Get an attribute by its ID."""
        for attr in self.attributes:
            if attr.attribute_id == attribute_id:
                return attr
        return None


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
