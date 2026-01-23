"""EPPI-specific data models extending the core models."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from deet.data_models.base import (
    Attribute,
    AttributeType,
    Document,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotation,
    ProcessedAnnotationData,
)


class EppiAttribute(Attribute):
    """
    EPPI-specific attribute with additional fields.

    Extends the core Attribute class with EPPI-specific
    metadata and hierarchy information.

    Uses alias generators to automatically map
    camelCase EPPI JSON fields to snake_case Python fields.
    """

    model_config = ConfigDict(alias_generators=to_camel)  # type: ignore[typeddict-unknown-key]

    # Core fields (inherited from Attribute) - these need manual processing
    question_target: str = ""  # Always empty for EPPI
    output_data_type: AttributeType = AttributeType.BOOL

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
    attribute_type: str | None = Field(
        description="Whether the attribute is Selectable in the "
        " EPPI-Reviewer interface or not",
        default=None,
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

    model_config = ConfigDict(alias_generators=to_camel)  # type: ignore[typeddict-unknown-key]

    # EPPI-specific fields - these map automatically from camelCase JSON
    item_id: int | None = None
    title: str | None = None
    parent_title: str | None = None
    short_title: str | None = None
    date_created: str | None = None
    edited_by: str | None = None
    year: str | None = None
    month: str | None = None
    abstract: str | None = None
    authors: str | None = None
    keywords: str | None = None
    doi: str | None = None


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
    GoldStandardAnnotatedDocument[EppiGoldStandardAnnotation]
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
