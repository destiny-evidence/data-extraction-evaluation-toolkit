"""Data models for the data extraction evaluation toolkit."""

from .base import (
    Attribute,
    AttributesList,
    Document,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotation,
)
from .eppi import EppiAttribute, EppiDocument, EppiGoldStandardAnnotation

__all__ = [
    "Attribute",
    "AttributesList",
    "Document",
    "EppiAttribute",
    "EppiDocument",
    "EppiGoldStandardAnnotation",
    "GoldStandardAnnotatedDocument",
    "GoldStandardAnnotation",
]
