"""
A register of supported supported annotation formats
and a map to their converters.
"""

from enum import StrEnum, auto

from deet.processors.base_converter import AnnotationConverter
from deet.processors.csv_annotation_converter import CSVAnnotationConverter
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter


class SupportedImportFormat(StrEnum):
    """Supported formats to import gold standard annotation data from."""

    EPPI_JSON = auto()
    GENERIC_CSV = auto()

    def get_annotation_converter(self) -> AnnotationConverter:
        """Return an instance of the parser for the given data type."""
        return CONVERTER_REGISTRY[self]()


# Registry mapping
CONVERTER_REGISTRY = {
    SupportedImportFormat.EPPI_JSON: EppiAnnotationConverter,
    SupportedImportFormat.GENERIC_CSV: CSVAnnotationConverter,
}
