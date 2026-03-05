"""
A register of supported supported annotation formats
and a map to their converters.
"""

from enum import StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deet.processors.base_converter import AnnotationConverter


class SupportedImportFormat(StrEnum):
    """Supported formats to import gold standard annotation data from."""

    EPPI_JSON = auto()

    def get_annotation_converter(
        self,
    ) -> "AnnotationConverter":
        """Return an instance of the parser for the given data type."""
        from deet.processors.eppi_annotation_converter import EppiAnnotationConverter

        mapping = {
            SupportedImportFormat.EPPI_JSON: EppiAnnotationConverter(),
        }
        return mapping[self]
