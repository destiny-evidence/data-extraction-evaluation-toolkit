"""Core data models for document processing and annotation."""

from __future__ import annotations

from typing import Any

from destiny_sdk.references import Reference
from pydantic import BaseModel


class Attribute(BaseModel):
    """
    Core attribute definition for data extraction tasks.

    Represents a single piece of information to be extracted from documents.
    """

    question_target: str  # 'How many patients were recruited?' - the prompt/question
    output_data_type: Any  # e.g. int, str, list, dict - expected data type
    attribute_id: str  # unique identifier for the attribute
    attribute_label: str  # human-readable way of identifying the attribute


class AttributesList(BaseModel):
    """Container for a list of attributes."""

    attributes: list[Attribute]

    def __iter__(self):
        """Make AttributesList iterable over its attributes."""
        for att in self.attributes:
            yield att

    def to_list(self) -> list[Attribute]:
        """Convert to a simple list of attributes."""
        return list(self)


class Document(BaseModel):
    """Represents a document in the dataset."""

    name: str
    citation: Reference
    context: str | list[str]
    document_id: str  # unique identifier for the document
    document_label: str  # human-readable way of identifying the document
    filename: str | None = None


class GoldStandardAnnotation(BaseModel):
    """A single gold standard annotation for an attribute."""

    attribute: Attribute
    output_data: Any


class GoldStandardAnnotatedDocument(BaseModel):
    """A document with its gold standard annotations."""

    document: Document
    annotations: list[GoldStandardAnnotation]
