"""Data models for procesed annotation data."""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic

from loguru import logger
from pydantic import BaseModel

from deet.data_models.base import (
    DEFAULT_ATTRIBUTE_TYPE,
    AnnotationType,
    Attribute,
    AttributeType,
    AttributeTypeVar,
    GoldStandardAnnotationTypeVar,
)
from deet.data_models.documents import (
    DocumentTypeVar,
    GoldStandardAnnotatedDocumentTypeVar,
)
from deet.data_models.enums import CustomPromptPopulationMethod
from deet.data_models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiRawData,
)


class ProcessedAttributeData(BaseModel, Generic[AttributeTypeVar]):
    """
    Structured result from annotation processing.

    Contains only attributes, so the ProcessedAnnotationData class can
    subclass this
    """

    attributes: list[AttributeTypeVar]

    def _custom_prompts_cli(self) -> None:
        """
        Use an interactive CLI to have the user enter custom prompts.

        Args:
            attribute (Attribute): a single (Eppi)Attribute

        """
        for attribute in self.attributes:
            attribute.enter_custom_prompt()

    def export_attributes_csv_file(self, filepath: Path) -> None:
        """
        Write a csv file containing all attributes for prompt population.

        Args:
            filepath (Path): outfile path.


        """
        if filepath.suffix != ".csv":
            bad_filetype = "file ending must be .csv"
            raise ValueError(bad_filetype)
        for attribute in self.attributes:
            attribute.write_to_csv(filepath=filepath)

        logger.info(f"wrote attributes to file {filepath}.")

    @staticmethod
    def _validate_csv_file(filepath: Path) -> None:
        """Validate csv file exists and has correct extension."""
        if not filepath.exists():
            no_file = f"CSV file not found: {filepath}"
            raise FileNotFoundError(no_file)

        if filepath.suffix != ".csv":
            bad_suffix = "File must have .csv extension"
            raise ValueError(bad_suffix)

    @staticmethod
    def _validate_csv_headers(fieldnames: Sequence[str] | None) -> None:
        """Validate csv has required headers."""
        if fieldnames is None:
            empty_csv = "csv file is empty or has no headers"
            raise ValueError(empty_csv)

        required_fields = ["attribute_id", "prompt"]
        for field in required_fields:
            if field not in fieldnames:
                csv_missing_fields = (
                    f"csv must contain '{field}' column. "
                    f"Found columns: {', '.join(fieldnames)}"
                )
                raise ValueError(csv_missing_fields)

    def _process_csv_row(
        self,
        row: dict[str, Any],
        csv_attribute_ids_with_prompts: set[int],
        *,
        overwrite: bool = True,
    ) -> bool:
        """
        Process a single csv row and update the matching attribute.

        Returns:
            bool: True if row was processed successfully, False otherwise

        """
        try:
            attribute_id = int(row.get("attribute_id"))  # type:ignore[arg-type]
        except ValueError as e:
            logger.warning(e)
            return False

        if (row.get("prompt") == "") or (row.get("prompt") is None):
            logger.debug(
                "prompt field is empty, "
                f"so we don't want this attribute {attribute_id}."
            )
            return False

        # Track attribute IDs that have non-empty prompts
        csv_attribute_ids_with_prompts.add(attribute_id)

        matching_attribute = None
        for attribute in self.attributes:
            if attribute.attribute_id == attribute_id:
                matching_attribute = attribute
                break

        if matching_attribute is None:
            logger.warning(f"No attribute found with ID {attribute_id}, skipping row")
            return False

        # Update attribute with prompt and data type
        try:
            matching_attribute.populate_prompt_from_dict(row, overwrite=overwrite)
            csv_attr_type = AttributeType(row.get("output_data_type"))  # type:ignore[arg-type]
            matching_attribute.output_data_type = csv_attr_type

        except ValueError as e:
            logger.error(
                f"Error processing row for attribute {attribute_id}: {e}"
                "setting attribute type to bool."
            )
            matching_attribute.output_data_type = DEFAULT_ATTRIBUTE_TYPE
            return False
        else:
            return True

    def _filter_attributes_by_csv(
        self,
        csv_attribute_ids_with_prompts: set[int],
        *,
        retain_only_csv_attributes: bool = True,
    ) -> None:
        """Filter attributes based on CSV content and retention policy."""
        if retain_only_csv_attributes:
            original_count = len(self.attributes)
            # Only keep attributes that are in CSV AND have non-empty prompts
            self.attributes = [
                attr
                for attr in self.attributes
                if attr.attribute_id in csv_attribute_ids_with_prompts
            ]
            logger.info(
                f"filtered attributes from {original_count} to {len(self.attributes)} "
                f"(retained only those in CSV with non-empty prompts)"
            )

    def _import_prompts_csv_file(
        self,
        filepath: Path,
        *,
        retain_only_csv_attributes: bool = True,
        overwrite: bool = True,
    ) -> None:
        """
        Import prompts from a csv file.

        Args:
            filepath (Path): attribute/prompt input file.
            retain_only_csv_attributes (bool, optional): if True, filter self.attributes
                to only include attributes with ids & a non-null prompt found in csv.
                Defaults to True.
            overwrite (bool, optional): Overwrite existing prompts. Defaults to True.

        """
        self._validate_csv_file(filepath)

        csv_attribute_ids_with_prompts: set[int] = set()
        rows_processed = 0

        with filepath.open(mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._validate_csv_headers(reader.fieldnames)

            for row in reader:
                if self._process_csv_row(
                    row=row,
                    csv_attribute_ids_with_prompts=csv_attribute_ids_with_prompts,
                    overwrite=overwrite,
                ):
                    rows_processed += 1

            logger.info(f"Processed {rows_processed} prompts from {filepath}")

        self._filter_attributes_by_csv(
            csv_attribute_ids_with_prompts=csv_attribute_ids_with_prompts,
            retain_only_csv_attributes=retain_only_csv_attributes,
        )

    def populate_custom_prompts(
        self,
        method: CustomPromptPopulationMethod,
        filepath: Path | None = None,
        **kwargs,
    ) -> None:
        """
        Populate custom prompts.

        Args:
            method (CustomPromptPopulationMethod)
            filepath (Path | None): infile path.

        Raises:
            FileNotFoundError: if method is file and there's no filepath.

        """
        if method == "cli":
            self._custom_prompts_cli()
        elif method == "file":
            if filepath is None:
                missing_filepath = "please specify a filepath!"
                raise FileNotFoundError(missing_filepath)
            self._import_prompts_csv_file(filepath=filepath, **kwargs)
        else:
            not_impl = f"method {method} is not implemented. use cli or file."
            raise NotImplementedError(not_impl)

    @property
    def total_attributes(self) -> int:
        """Total number of attributes processed."""
        return len(self.attributes)


class ProcessedAnnotationData(
    ProcessedAttributeData,
    Generic[
        AttributeTypeVar,
        DocumentTypeVar,
        GoldStandardAnnotationTypeVar,
        GoldStandardAnnotatedDocumentTypeVar,
    ],
):
    """
    Structured result from annotation processing.

    This model provides a clean, validated structure for all processed
    annotation data with useful properties and methods.
    """

    documents: list[DocumentTypeVar]
    annotations: list[GoldStandardAnnotationTypeVar]
    annotated_documents: list[GoldStandardAnnotatedDocumentTypeVar]
    attribute_id_to_label: dict[int, str]

    @property
    def total_documents(self) -> int:
        """Total number of documents processed."""
        return len(self.documents)

    @property
    def total_annotations(self) -> int:
        """Total number of annotations processed."""
        return len(self.annotations)

    @property
    def total_annotated_documents(self) -> int:
        """Total number of documents with annotations."""
        return len(self.annotated_documents)

    def get_attributes_by_attribute_type(
        self, attribute_type: AttributeType
    ) -> list[AttributeTypeVar]:
        """Get all attributes of a specific type."""
        return [
            attr for attr in self.attributes if attr.output_data_type == attribute_type
        ]

    def get_documents_with_annotations(self) -> list[DocumentTypeVar]:
        """Get only documents that have annotations."""
        annotated_doc_ids = {
            doc.document.document_id for doc in self.annotated_documents
        }
        return [doc for doc in self.documents if doc.document_id in annotated_doc_ids]

    def get_annotations_by_annotation_type(
        self, annotation_type: AnnotationType
    ) -> list[GoldStandardAnnotationTypeVar]:
        """Get all annotations of a specific type (human/llm)."""
        return [
            ann for ann in self.annotations if ann.annotation_type == annotation_type
        ]

    def get_attribute_by_id(self, attribute_id: int) -> Attribute | None:
        """Get an attribute by its ID."""
        for attr in self.attributes:
            if attr.attribute_id == attribute_id:
                return attr
        return None

    def export_linkage_mapper_csv(self, file_path: Path) -> None:
        """Export a csv mapper to link document IDs and filenames."""
        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["document_id", "name", "file_path"])
            writer.writeheader()
            for d in self.documents:
                d.init_document_identity()
                if d.document_identity is None:
                    message = f"document_identity was not set for document {d}"
                    raise ValueError(message)
                writer.writerow(
                    {
                        "document_id": d.document_identity.document_id,
                        "name": d.name,
                        "file_path": None,
                    }
                )


class ProcessedEppiAnnotationData(
    ProcessedAnnotationData[
        EppiAttribute,
        EppiDocument,
        EppiGoldStandardAnnotation,
        EppiGoldStandardAnnotatedDocument,
    ]
):
    """
    Structured result from EPPI annotation processing.

    This differs from Base ProcessedAnnotationData by specifying raw_data as an
    EppiRawData object
    """

    raw_data: EppiRawData | None = None
