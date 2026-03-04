"""EPPI-specific data models extending the core models."""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from destiny_sdk.enhancements import EnhancementFileInput, EnhancementType, Visibility
from destiny_sdk.parsers import EPPIParser
from destiny_sdk.parsers.exceptions import ExternalIdentifierNotFoundError
from destiny_sdk.references import ReferenceFileInput
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from deet.data_models.base import (  # ContextType,
    DEFAULT_ATTRIBUTE_TYPE,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
)
from deet.data_models.documents import (
    Document,
    GoldStandardAnnotatedDocument,
)

eppi_destiny_parser = EPPIParser(tags=["deet"])

DOI_REGEX = re.compile(
    r"(10\.\d{4,9}/[-._;()/:a-zA-Z0-9%<>\[\]+&]+)"
)  # for sanitising DOIs


def sanitise_doi(doi_candidate: str, *, raise_on_fail: bool = False) -> str:
    """Clean DOI strings in EPPI jsons."""
    doi = DOI_REGEX.search(doi_candidate)
    if doi and isinstance(doi, re.Match):
        return doi[0]
    if raise_on_fail:
        bad_doi = f"doi {doi} is bad."
        raise ValueError(bad_doi)
    logger.debug(
        "not found a valid DOI, returning empty string."
        " to modify this behaviour, set raise_on_fail=True"
    )
    return ""


def parse_citation_to_destiny(reference: dict[str, Any]) -> ReferenceFileInput:
    """
    Create a ReferenceFileInput object from document data.

    NOTE: we are not using the wrapping parser method in
    repository as it is for the whole document, and
    if it fails, we wouldn't be able to map a destiny reference.

    See https://github.com/destiny-evidence/destiny-repository/issues/458

    Args:
        reference: one reference from the eppi json.

    """
    if "DOI" in reference:
        reference["DOI"] = sanitise_doi(reference["DOI"])
    raw_enhancement_content = [
        c
        for c in [
            (
                eppi_destiny_parser._parse_abstract_enhancement(reference),  # noqa: SLF001
                EnhancementType.ABSTRACT,
            ),
            (
                eppi_destiny_parser._parse_bibliographic_enhancement(reference),  # noqa: SLF001
                EnhancementType.BIBLIOGRAPHIC,
            ),
            (
                eppi_destiny_parser._create_annotation_enhancement(),  # noqa: SLF001
                EnhancementType.ANNOTATION,
            ),
        ]
        if c[0] is not None
    ]

    enhancements = [
        EnhancementFileInput(
            source=eppi_destiny_parser.parser_source,
            visibility=Visibility.PUBLIC,
            content=content[0],  # type:ignore[arg-type]
            enhancement_type=content[1],  # type:ignore[call-arg]
        )
        for content in raw_enhancement_content
    ]

    destiny_ids = None
    try:
        destiny_ids = eppi_destiny_parser._parse_identifiers(  # noqa: SLF001
            ref_to_import=reference
        )
    except ExternalIdentifierNotFoundError as e:
        logger.warning(f"no identifier for reference. storing `None`. error: {e}")

    return ReferenceFileInput(
        visibility=Visibility.PUBLIC,
        identifiers=destiny_ids,
        enhancements=enhancements,
    )


class EppiAttributeSelectionType(StrEnum):
    """`AttributeType` as it appears in eppi json."""

    SELECTABLE = "Selectable (show checkbox)"
    OUTCOME = "Outcome"
    INTERVENTION = "Intervention"
    NOT_SELECTABLE = "Not Selectable (no checkbox)"

    @classmethod
    def _missing_(cls, value: object) -> "EppiAttributeSelectionType | None":
        """Handle case-insensitive assignment & lookup."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class EppiAttribute(Attribute):
    """
    EPPI-specific attribute with additional fields.

    Extends the core Attribute class with EPPI-specific
    metadata and hierarchy information.

    Uses alias generators to automatically map
    camelCase EPPI JSON fields to snake_case Python fields.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)  # type: ignore[typeddict-unknown-key]

    attribute_id: int = Field(
        validation_alias=AliasChoices("AttributeId", "attribute_id")
    )
    attribute_selection_type: EppiAttributeSelectionType = Field(
        validation_alias=AliasChoices(
            "AttributeType", "attribute_type", "attribute_selection_type"
        )
    )
    output_data_type: AttributeType = DEFAULT_ATTRIBUTE_TYPE
    attribute_label: str = Field(alias="AttributeName")

    # EPPI-specific fields - these map automatically from camelCase JSON
    attribute_set_description: str | None = Field(
        description="Description of the attribute set this attribute belongs to",
        default=None,
    )
    hierarchy_path: str | None = Field(
        description="Dot-separated path showing the hierarchical "
        " position of this attribute",
        default=None,
    )
    hierarchy_level: int = Field(
        description="Numeric level indicating depth in "
        " the attribute hierarchy (0 = root level)",
        default=0,
    )
    is_leaf: bool = Field(
        description="Whether this attribute is a leaf node  (has no child attributes)",
        default=True,
    )
    parent_attribute_id: int | None = Field(
        description="ID of the parent attribute in the hierarchy", default=None
    )
    attribute_description: str | None = Field(
        description="Detailed description explaining what this attribute represents",
        default=None,
    )


class EppiDocument(Document):
    """
    EPPI-specific document.

    Uses alias generators to automatically map
    camelCase EPPI JSON fields to snake_case Python fields.
    """

    name: str = Field(default="", validation_alias=AliasChoices("Title", "name"))
    document_id: int = Field(validation_alias=AliasChoices("ItemId", "document_id"))

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)  # type: ignore[typeddict-unknown-key]

    parent_title: str | None = Field(
        default=None, validation_alias=AliasChoices("ParentTitle", "parent_title")
    )
    short_title: str | None = Field(
        default=None, validation_alias=AliasChoices("ShortTitle", "short_title")
    )
    date_created: datetime | None = Field(
        default=None, validation_alias=AliasChoices("DateCreated", "date_created")
    )
    created_by: str | None = Field(
        default=None, validation_alias=AliasChoices("CreatedBy", "created_by")
    )
    edited_by: str | None = Field(
        default=None, validation_alias=AliasChoices("EditedBy", "edited_by")
    )
    year: int | None = Field(
        default=None, validation_alias=AliasChoices("Year", "year")
    )
    month: str | None = Field(
        default=None, validation_alias=AliasChoices("Month", "month")
    )
    abstract: str | None = Field(
        default=None, validation_alias=AliasChoices("Abstract", "abstract")
    )
    authors: str | None = Field(
        default=None, validation_alias=AliasChoices("Authors", "authors")
    )
    keywords: str | None = Field(
        default=None, validation_alias=AliasChoices("Keywords", "keywords")
    )
    doi: str | None = Field(default=None, validation_alias=AliasChoices("DOI", "doi"))

    @field_validator("date_created", mode="before")
    @classmethod
    def parse_date_string(cls, value: str) -> datetime | None | str:
        """Parse a string datetime to native datetime."""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # add as we encounter other formats, if ever relevant
            formats = [
                "%d/%m/%Y",  # OG EPPI
                "%Y-%m-%d %H:%M:%S%z",  # ISO format with timezone,
                # result of dumping is_final EppiDocument to json
                # MC: dumped data is not handled properly, but works if we let
                # pydantic handle it.
                "%Y-%m-%d",  # simple ISO date
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue

            return value

        return None

    @model_validator(mode="before")
    @classmethod
    def populate_citation_field(cls, data: dict[str, Any]) -> dict:
        """
        Populate the `citation` field with a Destiny
        reference derived from the EPPI data.
        """
        citation = data.get("citation")
        if citation is not None and ("enhancements" in citation):
            return data

        if isinstance(citation, ReferenceFileInput) and citation.enhancements:
            return data

        citation = parse_citation_to_destiny(reference=data)
        data["citation"] = citation

        return data


class EppiItemAttributeFullTextDetails(BaseModel):
    """
    EPPI-specific item attribute full text details.

    Arm specific information, exact text keywords for the attribute.
    """

    item_document_id: int | None = None
    text: str | None = None
    item_arm: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_at_least_one_field(cls, data: dict) -> dict:
        """Ensure at least one field is not None."""
        if all(v is None for k, v in data.items()):
            msg = (
                "At least one field must be provided "
                "(item_document_id, text, or item_arm)"
            )
            raise ValueError(msg)
        return data


class EppiGoldStandardAnnotation(GoldStandardAnnotation):
    """
    EPPI-specific gold standard annotation.

    In EPPI-Reviewer context, an "arm" refers to a study group or intervention group
    within a research study (e.g., "Treatment Group", "Control Group", "Placebo Group").
    Each annotation is associated with a specific arm to indicate which study group
    the extracted information relates to.
    """

    arm_id: int | None = Field(
        description="ID of the study arm this annotation relates to", default=None
    )
    arm_title: str | None = Field(
        description="Title or name of the study arm", default=None
    )
    arm_description: str | None = Field(
        description="Detailed description of the study arm", default=None
    )
    item_attribute_full_text_details: list[EppiItemAttributeFullTextDetails] | None = (
        Field(
            description="List of detailed text extracts and "
            " arm-specific information for this annotation",
            default=None,
        )
    )


class EppiGoldStandardAnnotatedDocument(
    GoldStandardAnnotatedDocument[EppiDocument, EppiGoldStandardAnnotation]
):
    """EPPI-specific gold standard annotated document."""


class EppiCodeSet(BaseModel):
    """
    Represents a single CodeSet from EPPI JSON.

    CodeSets contain hierarchical attribute definitions used in EPPI-Reviewer.
    """

    attributes: dict[str, Any] | None = Field(alias="Attributes", default=None)

    def get_attributes_list(self) -> list[dict[str, Any]]:
        """Extract AttributesList from the CodeSet."""
        if self.attributes and "AttributesList" in self.attributes:
            return self.attributes["AttributesList"]
        return []


class EppiRawData(BaseModel):
    """
    Represents the complete EPPI JSON structure.

    This model validates and structures the raw EPPI JSON data,
    making it easier to work with and validate.
    """

    code_sets: list[EppiCodeSet] = Field(alias="CodeSets", default=[])
    references: list[dict[str, Any]] = Field(alias="References", default=[])

    def extract_all_attributes(
        self, flatten_hierarchy_func: Callable[[list], list]
    ) -> list[dict[str, Any]]:
        """
        Extract and flatten attributes from all CodeSets.

        Args:
            flatten_hierarchy_func: Function to flatten attribute hierarchy

        Returns:
            List of flattened attribute dictionaries

        """
        all_attributes = []
        for codeset in self.code_sets:
            attributes_list = codeset.get_attributes_list()
            if attributes_list:
                flattened = flatten_hierarchy_func(attributes_list)
                all_attributes.extend(flattened)
        return all_attributes


class AttributeAnswerCoT(BaseModel):
    """Detailed answer format for a single attribute with reasoning."""

    attribute_name: str = Field(
        description="The name of the attribute being asked about"
    )
    answer: str = Field(description="The answer to the question, 'True' or 'False'")
    reasoning: str = Field(description="The reasoning behind the answer")
    citation: str | None = Field(
        description="The citation from the Research Information to support the answer"
    )


class BatchAnswerFormatCoT(BaseModel):
    """Batch answers for all attributes with reasoning."""

    answers: list[AttributeAnswerCoT] = Field(
        description="List of answers for each attribute"
    )
