"""EPPI-specific data models extending the core models."""

import re
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from destiny_sdk.enhancements import EnhancementFileInput, EnhancementType, Visibility
from destiny_sdk.parsers import EPPIParser
from destiny_sdk.references import ReferenceFileInput
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from deet.data_models.base import (
    Attribute,
    AttributeType,
    ContextType,
    Document,
    DocumentIDSource,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotation,
    ProcessedAnnotationData,
)

eppi_destiny_parser = EPPIParser(tags=["deet"])

DOI_REGEX = re.compile(
    r"(10\.\d{4,9}/[-._;()/:a-zA-Z0-9%<>\[\]+&]+)"
)  # for sanitising DOIs


def sanitise_doi(doi_candidate: str) -> str:
    """Clean DOI strings in EPPI jsons."""
    doi = DOI_REGEX.search(doi_candidate)
    if doi and isinstance(doi, re.Match):
        return doi[0]
    bad_doi = f"doi {doi} is bad."
    raise ValueError(bad_doi)


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
            content=content[0],  # the enhancement
            enhancement_type=content[1],  # the correct enhancement type
        )
        for content in raw_enhancement_content
    ]

    return ReferenceFileInput(
        visibility=Visibility.PUBLIC,
        identifiers=eppi_destiny_parser._parse_identifiers(  # noqa: SLF001
            ref_to_import=reference
        ),
        enhancements=enhancements,
    )


class EppiAttributeSelectionType(StrEnum):
    """`AttributeType` as it appears in eppi json."""

    SELECTABLE = "Selectable (show checkbox)"
    OUTCOME = "Outcome"
    INTERVENTION = "Intervention"
    NOT_SELECTABLE = "Not Selectable (no checkbox)"


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
    question_target: str = ""  # Always empty for EPPI
    output_data_type: AttributeType = AttributeType.BOOL
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
    # attribute_selection_type: EppiAttributeSelectionType | None = Field(
    #     description="Whether the attribute is Selectable in the "
    #     " EPPI-Reviewer interface or not",
    #     default=None,
    #     alias="AttributeType",
    # )
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
    context: str = ""
    context_type: ContextType = ContextType.EMPTY
    document_id: int = Field(validation_alias=AliasChoices("ItemId", "document_id"))
    document_id_source: DocumentIDSource = DocumentIDSource.EPPI_ITEM_ID

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)  # type: ignore[typeddict-unknown-key]

    # EPPI-specific fields - these map automatically from camelCase JSON
    # item_id: int
    # title: str
    parent_title: str | None = None
    short_title: str | None = None
    date_created: str | None = None
    edited_by: str | None = None
    year: str | None = None
    month: str | None = None
    abstract: str | None = None
    authors: str | None = None
    keywords: str | None = None
    doi: str | None = Field(default=None, validation_alias=AliasChoices("DOI", "doi"))

    @model_validator(mode="before")
    @classmethod
    def populate_citation_field(cls, data: dict[str, Any]) -> dict:
        """
        Populate the `citation` field with a Destiny
        reference derived from the EPPI data.
        """
        if not isinstance(data, dict):
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


class ProcessedEppiAnnotationData(
    ProcessedAnnotationData[
        EppiAttribute,
        EppiDocument,
        EppiGoldStandardAnnotation,
        EppiGoldStandardAnnotatedDocument,
    ]
):
    """
    Structured result from EPPI annotation processing.

    This differs from Base ProcessedAnnotationData by specifying raw_data as an
    EppiRawData object
    """

    raw_data: EppiRawData


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
