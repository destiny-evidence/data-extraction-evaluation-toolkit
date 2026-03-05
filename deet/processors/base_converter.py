"""Generic classes and functions for converters."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from pathlib import Path
from typing import Generic

from pydantic import TypeAdapter

from deet.data_models.base import AttributeType, AttributeTypeVar
from deet.data_models.documents import (
    DocumentTypeVar,
    GoldStandardAnnotatedDocumentTypeVar,
    GoldStandardAnnotationTypeVar,
)
from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.logger import logger

DEFAULT_BASE_OUTPUT_DIR = Path("tmp_parsed_eppi")
DEFAULT_ATTRIBUTES_FILENAME = "attributes.json"
DEFAULT_DOCUMENTS_FILENAME = "documents.json"
DEFAULT_ANNOTATED_DOCUMENTS_FILENAME = "annotated_documents.json"
DEFAULT_ATTRIBUTE_MAPPING_FILENAME = "attribute_id_to_label_mapping.json"


class Outfiles(StrEnum):
    """Enum of all outfiles producable by this module. Extend as required."""

    ATTRIBUTES = auto()
    DOCUMENTS = auto()
    ANNOTATED_DOCUMENTS = auto()
    ATTRIBUTE_LABEL_MAPPING = auto()


class AnnotationConverter(
    Generic[
        DocumentTypeVar,
        AttributeTypeVar,
        GoldStandardAnnotationTypeVar,
        GoldStandardAnnotatedDocumentTypeVar,
    ],
    ABC,
):
    """Abstract base class to define expected behaviour of an annotationconverter."""

    OUTFILE_LOADERS: dict[Outfiles, tuple[str, TypeAdapter]]

    @property
    def outfile_names(self) -> dict[Outfiles, str]:
        """Return a dictionary of outfiles to names, using outfile_loaders."""
        return {k: v[0] for k, v in self.OUTFILE_LOADERS.items()}

    def __init__(
        self,
        base_output_dir: str | Path | None = DEFAULT_BASE_OUTPUT_DIR,
    ) -> None:
        """
        Initialize the converter with configurable output paths.

        Args:
            output_dir: Base directory for saving processed files
            attributes_filename: Filename for attributes output
            documents_filename: Filename for documents output
            annotated_documents_filename: Filename for annotated documents output
            attribute_mapping_filename: Filename for attribute ID to label mapping

        """
        if base_output_dir is None:
            logger.debug(
                "`base_output_dir` set to None; "
                "converting to empty string for compatibility."
            )
            base_output_dir = ""
        self.base_output_dir = Path(base_output_dir)

    @property
    @abstractmethod
    def processed_data_type(
        self,
    ) -> type[
        ProcessedAnnotationData[
            AttributeTypeVar,
            DocumentTypeVar,
            GoldStandardAnnotationTypeVar,
            GoldStandardAnnotatedDocumentTypeVar,
        ]
    ]:
        """Return the class to use when instantiating processed data."""
        ...

    @abstractmethod
    def process_annotation_file(
        self,
        file_path: str | Path,
        set_attribute_type: str | AttributeType | None = None,
    ) -> ProcessedAnnotationData:
        """Parse a raw input format into processed data."""
        ...

    def write_processed_data_to_file(
        self,
        processed_data: ProcessedAnnotationData,
        output_dir: str | Path,
        outfiles_to_write: list[Outfiles] | None = None,
    ) -> dict[str, str]:
        """
        Save processed data to structured files using Pydantic model serialization.

        Args:
            processed_data: The processed data from process_annotation_file
            output_dir: Write all output (json) files from conversion to this
            directory. NOTE: we output files will live in a sub-directory
            `self.base_output_dir`, which by default is `DEFAULT_BASE_OUTPUT_DIR`.
            so, if you want output files to go straight to `output_dir`, set
            `self.base_output_dir` to '' or None.

        Returns:
            Dictionary mapping data types to saved file paths

        """
        file_mappings = {
            Outfiles.ATTRIBUTES: processed_data.attributes,
            Outfiles.DOCUMENTS: processed_data.documents,
            Outfiles.ANNOTATED_DOCUMENTS: processed_data.annotated_documents,
            Outfiles.ATTRIBUTE_LABEL_MAPPING: processed_data.attribute_id_to_label,
        }
        # setting here to avoid mutable default.
        if outfiles_to_write is None:
            outfiles_to_write = [Outfiles.ATTRIBUTES, Outfiles.DOCUMENTS]

        file_mappings = {
            k: v for k, v in file_mappings.items() if k in outfiles_to_write
        }
        logger.info(f"writing {",".join(k.value for k in file_mappings)} out...")

        user_dir = Path(output_dir)
        target_dir = user_dir / self.base_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"writing files to dir: {target_dir}")

        saved_files = {}

        for file_type, data_list in file_mappings.items():
            file_path = target_dir / self.outfile_names[file_type]
            logger.debug(f"writing file {file_type} to {file_path}")
            if file_type == Outfiles.ATTRIBUTE_LABEL_MAPPING:
                file_path.write_text(json.dumps(data_list))
            else:
                file_path.write_text(
                    json.dumps(
                        [item.model_dump(mode="json") for item in data_list],  # type: ignore[attr-defined]
                        indent=2,
                    )
                )
            saved_files[file_type.value] = str(file_path)

        return saved_files

    def reload_output(self, file_path: str | Path) -> ProcessedAnnotationData:
        """
        Read data back in, using OUTFILE_LOADERS and the
        subclass's processed_data_type.
        """
        loaded_data = {}
        for key, (filename, adapter) in self.OUTFILE_LOADERS.items():
            path = file_path / self.base_output_dir / filename
            if not path.exists():
                message = f"Expected {key.value} at {path}"
                raise FileNotFoundError(message)
            loaded_data[key] = adapter.validate_json(path.read_text())

        return self.processed_data_type(
            attributes=loaded_data[Outfiles.ATTRIBUTES],
            documents=loaded_data[Outfiles.DOCUMENTS],
            annotations=[],  # or derive from loaded_data if applicable
            annotated_documents=loaded_data[Outfiles.ANNOTATED_DOCUMENTS],
            attribute_id_to_label=loaded_data[Outfiles.ATTRIBUTE_LABEL_MAPPING],
        )
