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
    GENERIC_CSV = auto()

    def get_annotation_converter(
        self,
    ) -> "AnnotationConverter":
        """Return an instance of the parser for the given data type."""
        if self == SupportedImportFormat.EPPI_JSON:
            from deet.processors.eppi_annotation_converter import (
                EppiAnnotationConverter,
            )

            return EppiAnnotationConverter()
        if self == SupportedImportFormat.GENERIC_CSV:
            from deet.processors.csv_annotation_converter import (
                CSVAnnotationConverter,
            )

            return CSVAnnotationConverter()
        msg = f"No annotation converter implemented for {self}"
        raise ValueError(msg)
