"""
A register of supported supported annotation formats
and a map to their converters.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deet.processors.base_converter import AnnotationConverter

SUPPORTED_EXTENSIONS: set[str] = {".csv", ".json"}


class SupportedImportFormat(StrEnum):
    """Supported formats to import gold standard annotation data from."""

    EPPI_JSON = auto()
    GENERIC_CSV = auto()

    def get_annotation_converter(self) -> AnnotationConverter:
        """
        Return an instance of the converter for the given data type.

        Converters are imported lazily here, rather than at module load, because
        they pull in the heavy document/SDK stack. Keeping this module import-light
        keeps CLI startup fast for commands that only need the enum.
        """
        from deet.processors.csv_annotation_converter import (
            CSVAnnotationConverter,
        )
        from deet.processors.eppi_annotation_converter import (
            EppiAnnotationConverter,
        )

        registry: dict[SupportedImportFormat, type[AnnotationConverter]] = {
            SupportedImportFormat.EPPI_JSON: EppiAnnotationConverter,
            SupportedImportFormat.GENERIC_CSV: CSVAnnotationConverter,
        }
        return registry[self]()
