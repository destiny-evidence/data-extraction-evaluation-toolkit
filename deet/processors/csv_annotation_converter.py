"""Convert annotation CSV files to Pydantic models."""

import csv
import json
from collections import defaultdict
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from destiny_sdk.enhancements import (
    AbstractContentEnhancement,
    AbstractProcessType,
    AuthorPosition,
    Authorship,
    BibliographicMetadataEnhancement,
    EnhancementFileInput,
    Visibility,
)
from destiny_sdk.references import ReferenceFileInput
from pydantic import BaseModel

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
)
from deet.data_models.documents import (
    Document,
    GoldStandardAnnotatedDocument,
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
    """Enum of all outfiles producible by this module. Extend as required."""

    ATTRIBUTES = auto()
    DOCUMENTS = auto()
    ANNOTATED_DOCUMENTS = auto()
    ATTRIBUTE_LABEL_MAPPING = auto()


class CovidenceAnnotationConverter:
    """
    A class to convert raw Covidence CSV annotations
    into structured Pydantic models.
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
            Outfiles.ATTRIBUTE_LABEL_MAPPING: attribute_mapping_filename,
        }

    @staticmethod
    def normalize_text(value: str) -> str:
        """Strip white space and lowercase text."""
        return value.strip().lower()

    @staticmethod
    def find_duplicate_fieldnames(fieldnames: list) -> list[dict]:
        """
        Find duplicate column names and return their positions.

        Given a list of field/column names, returns a list of dictionaries
        containing the duplicated column name and the indices where it appears.

        Example:
            ["id", "name", "title", "name", "title"] ->
            [{"col_name": "name", "col_ids": [1, 3]},
             {"col_name": "title", "col_ids": [2, 4]}]

        """
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

    def infer_attribute_types(
        self,
        rows: list[dict],
        attribute_names: list[str],
        sample_size: int | None = None,
    ) -> dict[str, AttributeType]:
        """
        Infer field types for each column using up to `sample_size` non-null samples.
        Delfault sample size = entire dataset. Only use sample size if you are sure.
        This avoids sparse columns biasing the inference due to many empty values.

        Example:
        rows = [
            {"id": "1", "age": "42", "gender": "F"},
            {"id": "2", "age": "", "gender": "M"},
            {"id": "3", "age": "36", "gender": ""},
        ]
        fieldnames = ["id", "age", "gender"]

        Function:
        infer_field_types(rows, fieldnames, sample_size=2)

        Output:
        {"id": int, "age": int, "gender": str}

        """
        n_rows = len(rows)
        raw_attribute_types: defaultdict[str, list] = defaultdict(list)

        if not sample_size:
            sample_size = n_rows

        for att_name in attribute_names:
            row_num = 0

            while len(raw_attribute_types[att_name]) < sample_size and row_num < n_rows:
                att_type = self.infer_type(rows[row_num][att_name])

                if att_type is not None:
                    raw_attribute_types[att_name].append(att_type)

                row_num += 1

        resolved_attribute_types: dict[str, AttributeType] = {}
        for att_name, att_types in raw_attribute_types.items():
            try:
                resolved_attribute_types[att_name] = self.resolve_types(att_types)
            except ValueError as e:
                msg = f"Error inferring type for field '{att_name}': {e}"
                raise ValueError(msg) from e

        return resolved_attribute_types

    @staticmethod
    def _build_citation_dict_from_row(
        mapping: dict[str, str],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key, column_name in mapping.items():
            if column_name not in row:
                continue  # or raise, depending on strictness

            parts = key.split(".")
            d = result

            for part in parts[:-1]:
                d = d.setdefault(part, {})

            d[parts[-1]] = row[column_name].strip()

        return result

    @staticmethod
    def build_authorship_list(authors: str) -> list[Authorship]:
        """Create a Authorship list."""
        authorship = []
        author_list = authors.split(";")
        for i, author in enumerate(author_list):
            position = AuthorPosition.MIDDLE
            if i == 0:
                position = AuthorPosition.FIRST
            if i == len(author_list) - 1:
                position = AuthorPosition.LAST
            authorship.append(Authorship(display_name=author, position=position))
        return authorship

    def build_destiny_reference(
        self,
        row: dict,
        citation_fields: dict[str, Any],
        source: str = "DEET csv converter",
    ) -> ReferenceFileInput:
        """Convert citation dict to destiny reference."""
        citation_dict = self._build_citation_dict_from_row(citation_fields, row)

        abstract = (
            AbstractContentEnhancement(
                abstract=citation_dict["abstract"],
                process=AbstractProcessType.UNINVERTED,
            )
            if "abstract" in citation_dict
            else None
        )

        citation_dict["authorship"] = (
            self.build_authorship_list(citation_dict["authorship"])
            if "authorship" in citation_dict
            and isinstance(citation_dict["authorship"], str)
            else None
        )

        bibliographic_data = BibliographicMetadataEnhancement(**citation_dict)

        enhancements = [
            EnhancementFileInput(
                source=source,
                visibility=Visibility.PUBLIC,
                content=content,
            )
            for content in [abstract, bibliographic_data]
            if content is not None
        ]

        return ReferenceFileInput(
            visibility=Visibility.PUBLIC,
            enhancements=enhancements,
        )

    def load_csv(
        self,
        file_path: str | Path,
        attribute_fields: list[str] | None = None,
        citation_fields: dict | None = None,
    ) -> tuple[list[str], dict[str, str], list[str], list[dict[str, Any]]]:
        """Load CSV file, normalize fieldnames, return fieldnames, attributes, rows."""
        path = Path(file_path)
        with path.open(newline="") as f:
            csv_reader = csv.DictReader(f)

            # normalize headers BEFORE reading rows
            raw_headers = csv_reader.fieldnames or []
            fieldnames: list[str] = [self.normalize_text(h) for h in raw_headers]
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

            if citation_fields is None:
                logger.info("No citation fields provided")
                citation_fields = {}
            else:
                # normalize citation fields
                citation_fields = {
                    k: self.normalize_text(v) for k, v in citation_fields.items()
                }

                unknown = set(citation_fields.values()) - set(fieldnames)
                if unknown:
                    msg = f"Citation fields not found in CSV: {unknown}"
                    raise ValueError(msg)

            if attribute_fields is None:
                logger.info("No attribute fields provided")
                excluded_fields = meta_fields | set(citation_fields.values())
                resolved_attribute_fields = [
                    h for h in fieldnames if h not in excluded_fields
                ]
            else:
                resolved_attribute_fields = attribute_fields

            # Read rows into mem once
            rows = list(csv_reader)

        return fieldnames, citation_fields, resolved_attribute_fields, rows

    def build_attributes(
        self, attribute_types: dict[str, AttributeType]
    ) -> list[Attribute]:
        """Build and return an Attribute."""
        attributes = []
        for idx, (k, v) in enumerate(attribute_types.items()):
            attribute = Attribute(
                output_data_type=v,
                attribute_id=idx,
                attribute_label=k,
            )
            attributes.append(attribute)

        return attributes

    def build_documents_and_annotations(
        self,
        attributes: list[Attribute],
        citation_fields: dict,
        rows: list[dict],
    ) -> tuple[list[Document], list[GoldStandardAnnotatedDocument]]:
        """Build and return Documents and  Gold Standard Annotated Documents."""
        attr_by_label = {a.attribute_label: a for a in attributes}
        annotated_documents: list[GoldStandardAnnotatedDocument] = []
        documents: list[Document] = []

        for row_idx, row in enumerate(rows):
            row_reference = self.build_destiny_reference(row, citation_fields)

            # --- Build Document ---
            try:
                document = Document(
                    name=row["name"],
                    citation=row_reference,
                    document_id=row["document_id"],
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
                        f"field '{label}': {e}"
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
                document=document,  # <-- pass the model itself
                annotations=annotations,  # <-- list of annotation models
            )

            annotated_documents.append(annotated_doc)

        return documents, annotated_documents

    def process_annotation_file(
        self,
        file_path: str | Path,
        attribute_fields: list | None = None,
        citation_fields: dict | None = None,
    ) -> ProcessedAnnotationData:
        """
        Process a complete CSV annotation file and return structured data.

        Args:
            file_path: Path to the CSV annotation file

        Returns:
            ProcessedAnnotationData containing all processed data

        """
        logger.info(f"Processing annotation file: {file_path}")

        fieldnames, citation_fields, attribute_fields, rows = self.load_csv(
            file_path, attribute_fields, citation_fields
        )

        attribute_types = self.infer_attribute_types(rows, attribute_fields)

        attributes = self.build_attributes(attribute_types)

        documents, annotated_documents = self.build_documents_and_annotations(
            attributes, citation_fields, rows
        )
        attribute_id_to_label = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        logger.info(
            f"Processed {len(attributes)} attributes, {len(documents)} documents, "
            f" {len(annotated_documents)} annotated documents"
        )

        return ProcessedAnnotationData(
            attributes=attributes,
            documents=documents,
            annotated_documents=annotated_documents,
            attribute_id_to_label=attribute_id_to_label,
        )

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
