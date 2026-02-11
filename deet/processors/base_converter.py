"""Generic classes and functions for converters."""

from enum import StrEnum, auto

from deet.processors.eppi_annotation_converter import EppiAnnotationConverter


class SupportedImportFormat(StrEnum):
    """Supported formats to import gold standard annotation data from."""

    EPPI_JSON = auto()

    def get_annotation_converter(
        self,
    ) -> EppiAnnotationConverter:  # This should be a generic version
        """Return the parser for the given data type."""
        mapping = {SupportedImportFormat.EPPI_JSON: EppiAnnotationConverter()}
        return mapping[self]
