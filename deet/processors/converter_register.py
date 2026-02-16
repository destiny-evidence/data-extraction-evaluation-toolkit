"""
A register of supported supported annotation formats
and a map to their converters.
"""

from enum import StrEnum, auto

from deet.processors.base_converter import AnnotationConverter
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter


class SupportedImportFormat(StrEnum):
    """Supported formats to import gold standard annotation data from."""

    EPPI_JSON = auto()
    DEET = auto()

    def get_annotation_converter(
        self,
    ) -> AnnotationConverter:
        """Return the parser for the given data type."""
        mapping = {
            SupportedImportFormat.EPPI_JSON: EppiAnnotationConverter(),
            SupportedImportFormat.DEET: AnnotationConverter(),
        }
        return mapping[self]
