"""Core data models for document processing and annotation."""

import csv
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from random import randint
from typing import Any, Literal

from destiny_sdk.references import ReferenceFileInput
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tabulate import tabulate

from deet.exceptions import BadDocumentIdError, MissingCitationElementError
from deet.utils.identifier_utils import (
    DOCUMENT_ID_N_DIGITS,
    MAX_DOCUMENT_ID,
    MIN_DOCUMENT_ID,
    hash_n_strings_to_eight_digit_int,
)

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
    DOI_AUTHOR_YEAR = auto()
    DOI_ID = auto()
    AUTHOR_YEAR_ID = auto()
    RANDINT = auto()


class DocumentIdentity(BaseModel):
    """A unified identity for a document, deriveable from multiple sources."""

    document_id: int | None = None
    document_id_source: DocumentIDSource | None = None

    # parsed citation info
    citation_dict: dict  # a dump of the reference as dict, via ReferencePresenter

    def populate_id(
        self,
        existing_ids: list[int] | None = None,
        hierarchy: list[DocumentIDSource] | None = None,
    ) -> None:
        """
        Populate document_id using a hierarchical list of ID creation methods.

        Tries each method in order until a unique ID is generated. If an ID
        conflicts with existing_ids, tries the next method. RANDINT always
        succeeds as fallback.

        NOTE: we will have to implement some sort of matching thing, if we are
        concerned that an id-collision might be becuase we have already
        parsed&linked a document.

        Args:
            existing_ids: List of existing IDs to check for conflicts.
            hierarchy: Ordered list of DocumentIDSource methods to try.
                Defaults to [EPPI_ITEM_ID, DOI_ID, DOI_AUTHOR_YEAR,
                AUTHOR_YEAR_ID, RANDINT].

        Raises:
            BadDocumentIdError: If unable to generate unique ID (should never
                happen as RANDINT is always in hierarchy).

        """
        if existing_ids is None:
            existing_ids = []

        if hierarchy is None:
            hierarchy = list(DocumentIDSource)

        if DocumentIDSource.RANDINT not in hierarchy:
            hierarchy.append(DocumentIDSource.RANDINT)

        attempted_sources = []

        for id_source in hierarchy:
            try:
                id_factory = self._create_id_factory(id_source)
                logger.debug(f"created id_factory: {id_factory}")
                potential_id = id_factory()
                logger.debug(f"created potential id: {potential_id}")

                # id collisions?
                if potential_id not in existing_ids:
                    self.document_id = potential_id
                    self.document_id_source = id_source
                    logger.debug(
                        f"successfully created document_id {potential_id} "
                        f"using {id_source}"
                    )
                    return

                logger.debug(
                    f"id {potential_id} from {id_source} conflicts with existing IDs"
                )
                attempted_sources.append(id_source)

            except (BadDocumentIdError, MissingCitationElementError) as e:
                logger.debug(f"Failed to create ID using {id_source}: {e}")
                attempted_sources.append(id_source)
                continue

        failed_sources = ", ".join(str(s) for s in attempted_sources)
        err_msg = (
            f"Failed to generate unique document_id after trying: {failed_sources}"
        )

        if len(attempted_sources) == len(hierarchy):
            max_attempts = 10
            attempts = 0
            while True:
                potential_id = self._random_int_id()
                if potential_id not in existing_ids:
                    self.document_id = potential_id
                    self.document_id_source = DocumentIDSource.RANDINT
                    logger.debug(
                        f"successfully created document_id {potential_id} "
                        f"using {id_source}"
                    )
                    return
                attempts += 1
                if attempts == max_attempts:
                    err_msg += f" plus {attempts} randint attempts."
                    break

        raise BadDocumentIdError(err_msg)

    def _create_id_factory(self, id_source: DocumentIDSource) -> Callable:
        """
        Return an id-creating method given specific value of DocumentIDSource.

        Returns:
            int: the id.

        """
        id_creation_map = {
            DocumentIDSource.EPPI_ITEM_ID: self._eppi_item_id,
            DocumentIDSource.DOI_ID: self._doi_id,
            DocumentIDSource.AUTHOR_YEAR_ID: self._author_year_id,
            DocumentIDSource.DOI_AUTHOR_YEAR: self._doi_author_year_id,
            DocumentIDSource.RANDINT: self._random_int_id,
        }

        return id_creation_map[id_source]

    def _eppi_item_id(self) -> None:
        """Map an existing item_id (parsed as document_id)."""
        # we're going to assume that our `document_id`, received
        # from parsing eppi-json to EppiDocument is always going
        # to be eppi, otherwise this method should be extended to
        # reflect it coming from somewhere else.
        # either way, it'll have to be an 8-digit int.
        if (
            self.document_id is not None
            and isinstance(self.document_id, int)
            and len(str(self.document_id)) == DOCUMENT_ID_N_DIGITS
        ):
            return
        bad_doc_id = f"id {self.document_id} is not a valid eppi item_id."
        raise BadDocumentIdError(bad_doc_id)

    def _citation_id_hasher(self, target_fields: list[str]) -> int:
        """Create an id from _n_ citation fields."""
        if not all(field in self.citation_dict for field in target_fields):
            missing_citation = (
                f"required fields are missing in citation. "
                "required: {', '.join(target_fields)}"
                f"actual: {','.join(self.citation_dict)}"
            )
            raise MissingCitationElementError(missing_citation)
        payload = [self.citation_dict[field] for field in target_fields]
        if "" in payload or None in payload:
            none_or_empty = (
                "some or all of target fields are "
                f"None or empty strings: {','.join(target_fields)} "
            )
            raise MissingCitationElementError(none_or_empty)
        return hash_n_strings_to_eight_digit_int(payload)

    def _doi_id(self) -> int:
        """Create an integer id as a function of doi."""
        return self._citation_id_hasher([self.citation_dict["doi"]])

    def _doi_author_year_id(self) -> int:
        """Create an integer id as a function of doi, author and year."""
        return self._citation_id_hasher(
            [
                self.citation_dict["doi"],
                self.citation_dict["author"],
                self.citation_dict["year"],
            ]
        )

    def _author_year_id(self) -> int:
        """Create an 8-digit integer id as a function of author and year."""
        return self._citation_id_hasher(
            [
                self.citation_dict["author"],
                self.citation_dict["year"],
            ]
        )

    @staticmethod
    def _random_int_id() -> int:
        """Create a random integer id with 8 digits."""
        return randint(MIN_DOCUMENT_ID, MAX_DOCUMENT_ID)  # noqa: S311


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
    context: str | None = None  # new defaults, empty
    context_type: ContextType | None = None
    document_id: int | None = None
    document_id_source: DocumentIDSource | None = None
    document_identity: DocumentIdentity | None = None
    document_filepath: Path | None = None

    def init_document_identity(self):
        # dump the citation to dict, write the id stuff if exists.
        pass

    def link_parsed_pdf(self, parsed_document: str):
        # attach the string output from the parser to `context` field
        # update the `context_type`
        # update document_filepath
        pass

    def update_document_identity(self):
        # update the remaining empty fields
        pass


class LinkedDocument(Document):
    """A document linked to an actual context/body, usually derived from a pdf."""

    context: str
    context_type: ContextType
    document_id: int
    document_id_source: DocumentIDSource
    document_identity: DocumentIdentity
    document_filepath: Path = None

    def pickle(self, path: Path) -> None:
        """Pickle a fully populated linked document to file."""
        raise NotImplementedError


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


class GoldStandardAnnotatedDocument(Document):
    """A document with its gold standard annotations."""

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
