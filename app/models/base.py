"""Core data models for document processing and annotation."""

from enum import Enum
from typing import Any

from destiny_sdk.references import Reference
from pydantic import BaseModel


class AnnotationType(str, Enum):
    """Enumeration of annotation types."""

    HUMAN = "human"
    LLM = "llm"


class Attribute(BaseModel):
    """
    Core attribute definition for data extraction tasks.

    Represents a single piece of information to be extracted from documents.
    """

    question_target: str  # 'How many patients were recruited?' - the prompt/question
    output_data_type: str  # Expected data type for the attribute (e.g., "bool", "int", "str")
    attribute_id: str  # unique identifier for the attribute
    attribute_label: str  # human-readable way of identifying the attribute


class AttributesList(BaseModel):
    """Container for a list of attributes."""

    attributes: list[Attribute]

    def __iter__(self):  # noqa: ANN204
        """Make AttributesList iterable over its attributes."""
        yield from self.attributes

    def to_list(self) -> list[Attribute]:
        """Convert to a simple list of attributes."""
        return list(self)


class Document(BaseModel):
    """Represents a document in the dataset."""

    name: str
    citation: Reference
    context: str | list[str]
    document_id: str
    filename: str | None = None


class GoldStandardAnnotation(BaseModel):
    """A single gold standard annotation for an attribute."""

    attribute: Attribute
    output_data: Any
    annotation_type: AnnotationType


class GoldStandardAnnotatedDocument(Document):
    """A document with its gold standard annotations."""

    annotations: list[GoldStandardAnnotation]
