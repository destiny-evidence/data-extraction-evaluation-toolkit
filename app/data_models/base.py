"""Core data models for document processing and annotation."""

from enum import StrEnum, auto
from typing import Any

from destiny_sdk.references import Reference
from pydantic import BaseModel


class AnnotationType(StrEnum):
    """Enumeration of annotation types."""

    HUMAN = auto()
    LLM = auto()


class AttributeType(StrEnum):
    """Enum for permitted attribute data types."""

    BOOL = auto()
    INT = auto()
    LIST = auto()
    DICT = auto()
    FLOAT = auto()


class Attribute(BaseModel):
    """
    Core attribute definition for data extraction tasks.

    Represents a single piece of information to be extracted from documents.
    """

    question_target: str  # 'How many patients were recruited?' - the prompt/question
    output_data_type: AttributeType
    attribute_id: int  # unique identifier for the attribute
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
