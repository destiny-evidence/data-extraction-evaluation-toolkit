"""Convert annotation CSV files to Pydantic models."""

# - Esure there are no duplicate column names else Raise an error.
# - Require for 2 columns = name and doc_id.
# - infer datatypes based on sample of 100 csv lines? -- majority wins? --coarse others?
# - how to handle NA, none, not reported etc?
# - allow the option for users to provide types?

import csv
import json
import random
from collections import defaultdict
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from destiny_sdk.enhancements import Visibility
from pydantic import BaseModel

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    ContextType,
    Document,
    DocumentIDSource,
    GoldStandardAnnotatedDocument,
    GoldStandardAnnotation,
    ReferenceFileInput,
)
from deet.logger import logger

DEFAULT_BASE_OUTPUT_DIR = Path("tmp_parsed_covidence")
DEFAULT_ATTRIBUTES_FILENAME = "attributes.json"
DEFAULT_DOCUMENTS_FILENAME = "documents.json"
DEFAULT_ANNOTATED_DOCUMENTS_FILENAME = "annotated_documents.json"
DEFAULT_ATTRIBUTE_MAPPING_FILENAME = "attribute_id_to_label_mapping.json"

DEFAULT_SAMPLE_SIZE = 100


class ProcessedAnnotationData(BaseModel):
    """Structured result from annotation processing."""

    attributes: list[Attribute]
    documents: list[Document]
    annotated_documents: list[GoldStandardAnnotatedDocument]
    attribute_id_to_label: dict[int, str]


class Outfiles(StrEnum):
    """Enum of all outfiles producable by this module. Extend as required."""

    ATTRIBUTES = auto()
    DOCUMENTS = auto()
    ANNOTATED_DOCUMENTS = auto()
    ATTRIBUTE_LABEL_MAPPING = auto()


class CovidenceAnnotationConverter:
    """
    A class to convert raw Covidence CSV annotations
    into structured Pydantic models.

    This converter handles the complex hierarchical
    structure of EPPI attributes by flattening
    them while preserving parent-child relationships
    through path information.
    """

    def __init__(
        self,
        base_output_dir: str | Path | None = DEFAULT_BASE_OUTPUT_DIR,
        attributes_filename: str = DEFAULT_ATTRIBUTES_FILENAME,
        documents_filename: str = DEFAULT_DOCUMENTS_FILENAME,
        annotated_documents_filename: str = DEFAULT_ANNOTATED_DOCUMENTS_FILENAME,
        attribute_mapping_filename: str = DEFAULT_ATTRIBUTE_MAPPING_FILENAME,
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

        # extend below if adding more output files in `Outfiles`.
        self.outfilename_object_map = {
            Outfiles.ATTRIBUTES: attributes_filename,
            Outfiles.DOCUMENTS: documents_filename,
            Outfiles.ANNOTATED_DOCUMENTS: annotated_documents_filename,
            # Outfiles.ATTRIBUTE_LABEL_MAPPING: attribute_mapping_filename,
        }

    def find_duplicate_fieldnames(self, fieldnames: list) -> list:
        """Return duplicate column names and their index."""
        positions = defaultdict(list)
        for idx, name in enumerate(fieldnames):
            positions[name].append(idx)

        return [
            {"col_name": name, "col_ids": ids}
            for name, ids in positions.items()
            if len(ids) > 1
        ]

    def infer_type(self, value: str) -> type | None:
        """Infer value types."""
        if value is None or str(value).strip() == "":
            return None

        v = str(value).strip().lower()

        if v in {"true", "false"}:
            return bool
        try:
            float(v)
            if "." in v:
                return float
            else:  # noqa: RET505
                return int
        except ValueError:
            return str

    # TODO: need to add list and dictionary handling
    def resolve_types(self, types: list) -> AttributeType:
        """Given a list of types, resolve to a single type."""
        unique = set(types) - {None}
        if len(unique) == 0:
            msg = "Type not inferred. Sample is null."
            raise ValueError(msg)
        if unique == {str}:
            return AttributeType.STRING
        if unique == {bool}:
            return AttributeType.BOOL
        if unique == {int}:
            return AttributeType.INTEGER
        if unique.issubset({float, int}) and bool not in unique:
            return AttributeType.FLOAT
        msg = "Multiple incompatible types found"
        raise ValueError(msg)

    def infer_field_types(
        self,
        rows: list[dict],
        fieldnames: list[str],
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> dict[str, AttributeType]:
        """Infer field types for the given columns."""
        sample_size = min(len(rows), sample_size)

        rng = random.Random(42)  # noqa: S311
        sample_rows = rng.sample(rows, sample_size)

        raw_fieldtypes = defaultdict(list)
        for row in sample_rows:
            for fname in fieldnames:
                ftype = self.infer_type(row[fname])
                raw_fieldtypes[fname].append(ftype)

        fieldtypes: dict[str, AttributeType] = {}
        for fname, ftypes in raw_fieldtypes.items():
            try:
                fieldtypes[fname] = self.resolve_types(ftypes)
            except ValueError as e:
                msg = f"Error inferring type for field '{fname}': {e}"
                raise ValueError(msg) from e

        return fieldtypes

    def load_csv(
        self, file_path: str | Path
    ) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        """Load CSV file, normalize fieldnames, return fieldnames, attributes, rows."""
        path = Path(file_path)
        with path.open(newline="") as f:
            csv_reader = csv.DictReader(f)

            # normalize headers BEFORE reading rows
            raw_headers = csv_reader.fieldnames or []
            fieldnames = [h.strip().lower() for h in raw_headers]
            csv_reader.fieldnames = fieldnames

            # validate duplicates
            dup_fields = self.find_duplicate_fieldnames(fieldnames)
            if dup_fields:
                msg = f"{len(dup_fields)} Duplicate fieldnames found: {dup_fields}"
                raise ValueError(msg)

            # validate required fields
            meta_fields = {"name", "document_id"}
            missing = meta_fields - set(fieldnames)
            if missing:
                msg = f"Required columns missing: {missing}"
                raise ValueError(msg)

            # attribute fields
            attribute_fields: list[str] = [
                h for h in fieldnames if h not in meta_fields
            ]

            # Read rows into mem once
            rows = list(csv_reader)

        return fieldnames, attribute_fields, rows

    def build_attributes(
        self, attribute_types: dict[str, AttributeType]
    ) -> list[Attribute]:
        """Build and return an Attribute."""
        attributes = []
        for idx, (k, v) in enumerate(attribute_types.items()):
            attribute = Attribute(
                question_target=k,
                output_data_type=v,
                attribute_id=idx,
                attribute_label=k,
            )
            attributes.append(attribute)

        return attributes

    def build_documents_and_annotations(
        self,
        rows: list[dict],
        attributes: list[Attribute],
    ) -> tuple[list[Document], list[GoldStandardAnnotatedDocument]]:
        """Build and return Documents and  Gold Standard Annotated Documents."""
        documents: list[Document] = []
        annotated_documents: list[GoldStandardAnnotatedDocument] = []

        attr_by_label = {a.attribute_label: a for a in attributes}
        for row_idx, row in enumerate(rows):
            # --- Build Document ---
            try:
                document = Document(
                    name=row["name"],
                    citation=ReferenceFileInput(visibility=Visibility.PUBLIC),
                    context="",
                    context_type=ContextType.EMPTY,
                    document_id=row["document_id"],
                    document_id_source=DocumentIDSource.CSV_ITEM_ID,
                )
            except KeyError as e:
                msg = f"Missing required document field {e} in row {row_idx}"
                raise KeyError(msg) from e

            documents.append(document)

            # --- Build Annotations ---
            annotations: list[GoldStandardAnnotation] = []
            for label, attr in attr_by_label.items():
                raw_value = row.get(label)
                python_type = attr.output_data_type.to_python_type()

                try:
                    converted_value = python_type(raw_value)
                except (TypeError, ValueError) as e:
                    msg = (
                        f"Type conversion failed for row {row_idx}, "
                        "field '{label}': {e}"
                    )
                    raise ValueError(msg) from e

                annotation = GoldStandardAnnotation(
                    attribute=attr,
                    output_data=converted_value,
                    annotation_type=AnnotationType.HUMAN,
                )

            annotations.append(annotation)

            # --- Attach annotations to document ---
            annotated_doc = GoldStandardAnnotatedDocument(
                **document.model_dump(),
                annotations=annotations,
            )

            annotated_documents.append(annotated_doc)

        return documents, annotated_documents

    def process_annotation_file(self, file_path: str | Path) -> ProcessedAnnotationData:
        """
        Process a complete annotation file and return structured data.

        Args:
            file_path: Path to the JSON annotation file

        Returns:
            ProcessedAnnotationData containing all processed data

        """
        logger.info(f"Processing annotation file: {file_path}")

        fieldnames, attribute_fields, rows = self.load_csv(file_path)

        attribute_types = self.infer_field_types(rows, attribute_fields)

        attributes = self.build_attributes(attribute_types)

        documents, annotated_documents = self.build_documents_and_annotations(
            rows, attributes
        )
        attribute_id_to_label = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        return ProcessedAnnotationData(
            attributes=attributes,
            documents=documents,
            annotated_documents=annotated_documents,
            attribute_id_to_label=attribute_id_to_label,
        )

    #     logger.info(
    #         f"Processed {len(attributes)} attributes, {{len(documents)}} documents, "
    #         # f"{len(all_annotations)} annotations,"
    #         # " {len(annotated_documents)} annotated documents"
    #     )

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
        logger.info(f"writing {','.join(file_mappings.keys())} out...")

        user_dir = Path(output_dir)
        target_dir = user_dir / self.base_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"writing files to dir: {target_dir}")

        saved_files = {}

        for file_type, data_list in file_mappings.items():
            file_path = target_dir / self.outfilename_object_map[file_type]
            logger.debug(f"writing file {file_type} to {file_path}")
            file_path.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in data_list],  # type: ignore[attr-defined]
                    indent=2,
                )
            )
            saved_files[file_type.value] = str(file_path)

        return saved_files
