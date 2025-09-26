"""EPPI-specific data models extending the core models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import Attribute, Document


class EppiAttribute(Attribute):
    """
    EPPI-specific attribute with additional fields.

    Extends the core Attribute class with EPPI-specific metadata and hierarchy information.
    """

    attribute_set_description: str | None = None
    hierarchy_path: str | None = None
    hierarchy_level: int = 0
    is_leaf: bool = True
    parent_attribute_id: str | None = None
    attribute_type: str | None = None
    attribute_description: str | None = None


class EppiDocument(Document):
    """EPPI-specific document."""

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


class EppiGoldStandardAnnotation(BaseModel):
    """EPPI-specific gold standard annotation."""

    attribute: EppiAttribute
    additional_text: str | None = None
    arm_id: int | None = None
    arm_title: str | None = None
    arm_description: str | None = None
    output_data: bool | None = True
    item_attribute_full_text_details: list[EppiItemAttributeFullTextDetails] | None = (
        None
    )


class EppiGoldStandardAnnotatedDocument(BaseModel):
    """EPPI-specific gold standard annotated document."""

    document: EppiDocument
    annotations: list[EppiGoldStandardAnnotation]


class AttributeAnswerCoT(BaseModel):
    """Detailed answer format for a single attribute with reasoning."""

    attribute_name: str = Field(
        description="The name of the attribute being asked about"
    )
    Answer: str = Field(description="The answer to the question, 'True' or 'False'")
    Reasoning: str = Field(description="The reasoning behind the answer")
    Citation: str | None = Field(
        description="The citation from the Research Information to support the answer"
    )


class BatchAnswerFormatCoT(BaseModel):
    """Batch answers for all attributes with reasoning."""

    answers: list[AttributeAnswerCoT] = Field(
        description="List of answers for each attribute"
    )
