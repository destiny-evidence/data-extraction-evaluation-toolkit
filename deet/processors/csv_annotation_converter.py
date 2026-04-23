"""Convert annotation CSV files to Pydantic models."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
from xmlrpc.client import Boolean

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
from loguru import logger
from pydantic import BaseModel, TypeAdapter

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
from deet.data_models.processed_gold_standard_annotations import (
    ProcessedAnnotationData,
)
from deet.processors.base_converter import (
    DEFAULT_ANNOTATED_DOCUMENTS_FILENAME,
    DEFAULT_ATTRIBUTE_MAPPING_FILENAME,
    DEFAULT_ATTRIBUTES_FILENAME,
    DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_DOCUMENTS_FILENAME,
    AnnotationConverter,
    Outfiles,
)

ALLOWED_REFERENCE_MAPPING_KEYS = {
    "abstract",
    "authorship",
    "cited_by_count",
    "created_date",
    "updated_date",
    "publication_date",
    "publication_year",
    "publisher",
    "title",
    "pagination.volume",
    "pagination.issue",
    "pagination.first_page",
    "pagination.last_page",
    "publication_venue.display_name",
    "publication_venue.issn",
    "publication_venue.venue_type",
    "publication_venue.issn_l",
    "publication_venue.host_organization_name",
}


class ColumnTypeInferenceError(Exception):
    """Raised when column type inference fails due to incompatible types."""


class CSVParserConfig(BaseModel):
    """Configuration Seetings for parsing CSV."""

    author_separator: str = ";"
    auto_assign_reference_fields: Boolean = False


class CSVAnnotationConverter(AnnotationConverter):
    """
    A class to convert raw CSV (e.g. Covidence) annotations
    into structured Pydantic models.

     This converter operates on flat CSV columns, infers field/column types,
    and produces attributes, documents, and annotated document records.
    """

    def __init__(  # noqa: PLR0913
        self,
        base_output_dir: str | Path | None = DEFAULT_BASE_OUTPUT_DIR,
        attributes_filename: str = DEFAULT_ATTRIBUTES_FILENAME,
        documents_filename: str = DEFAULT_DOCUMENTS_FILENAME,
        annotated_documents_filename: str = DEFAULT_ANNOTATED_DOCUMENTS_FILENAME,
        attribute_mapping_filename: str = DEFAULT_ATTRIBUTE_MAPPING_FILENAME,
        config: CSVParserConfig | None = None,
    ) -> None:
        """
        Initialize the converter output configurations (base directory + filenames) to
        save the multiple files created during csv processing.

        Args:
            base_output_dir: Base directory for saving processed files
            attributes_filename: Filename for attributes output
            documents_filename: Filename for documents output
            annotated_documents_filename: Filename for annotated documents output
            attribute_mapping_filename: Filename for attribute ID to label mapping

        """
        self.config = config or CSVParserConfig()
        # If no directory given, write everything relative to current working directory
        if base_output_dir is None:
            logger.debug(
                "`base_output_dir` set to None; "
                "converting to empty string for compatibility."
            )
            base_output_dir = ""
        self.base_output_dir = Path(base_output_dir)

        # Output registry:
        # For each output type, provides the filename and how to serialize/validate it.
        # Used when writing and reloading the results

        self.OUTFILE_LOADERS: dict[Outfiles, tuple[str, TypeAdapter]] = {
            Outfiles.ATTRIBUTES: (
                attributes_filename,
                TypeAdapter(list[Attribute]),
            ),
            Outfiles.DOCUMENTS: (
                documents_filename,
                TypeAdapter(list[Document]),
            ),
            Outfiles.ANNOTATED_DOCUMENTS: (
                annotated_documents_filename,
                TypeAdapter(list[GoldStandardAnnotatedDocument]),
            ),
            Outfiles.ATTRIBUTE_LABEL_MAPPING: (
                attribute_mapping_filename,
                TypeAdapter(dict[int, str]),
            ),
        }

    @property
    def processed_data_type(self) -> type[ProcessedAnnotationData]:
        """Return ProcessedAnnotationData."""
        return ProcessedAnnotationData

    @staticmethod
    def _find_duplicate_column_names(column_names: list[str]) -> list[dict]:
        """
        Find duplicate column names and return their positions.

        Given a list of field/column names, returns a list of dictionaries
        containing the duplicated column name and the indices where it appears.

        Example:
            column_names = ["id", "name", "title", "name", "title"] ->
            {"name": [1, 3], "title": [2, 4]}

        """
        positions = defaultdict(list)
        for idx, name in enumerate(column_names):
            positions[name].append(idx)

        return [{name: ids} for name, ids in positions.items() if len(ids) > 1]

    @staticmethod
    def _infer_type(value: str) -> type | None:
        """
        Given a string value infer its type.
        Returns bool, int, float, or str depending on the value.
        Empty or null-like values return None.
        """
        if value is None or str(value).strip() == "":
            return None

        v = str(value).strip().lower()

        if v in {"true", "false", "t", "f"}:
            return bool
        try:
            float(v)
            if v.lstrip("-+").isdigit():
                return int
            return float  # noqa: TRY300
        except ValueError:
            return str

    # TODO: Extend support for list and dict AttributeType values.
    # Discussed use case: 'tags' associated with a document ([USA, Germnay, India])

    @staticmethod
    def _resolve_types(types: list) -> AttributeType:
        """
        Given a list of types, resolve to a single type and return an
        AttributeType object.

        Examples:
        [int, int, None, int, None, int] -> int -> AttributeType.INTEGER
        [int, int, None, float, None, int] -> float -> AttributeType.FLOAT

        """
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

        msg = f"Multiple incompatible types found: {unique}"
        raise ColumnTypeInferenceError(msg)

    def _infer_column_types(
        self,
        rows: list[dict],
        column_names: list[str],
        sample_size: int | None = None,
    ) -> dict[str, AttributeType]:
        """
        Infer the data type of each column using up to `sample_size` non-null samples.

        By default, `sample_size` is the total number of rows. Specifying a smaller
        sample can speed up inference but may bias results if columns are sparse.

        Args:
            rows: List of dictionaries representing CSV rows.
            column_names: List of column names for which to infer data types.
            sample_size: Maximum number of non-null samples to use per column.

        Returns:
        - Dictionary mapping each column name to its inferred `AttributeType`.

        Example:
            rows = [
                {"id": "1", "age": "42", "gender": "F"},
                {"id": "2", "age": "", "gender": "M"},
                {"id": "3", "age": "36", "gender": ""},
            ]
            fieldnames = ["id", "age", "gender"]

            converter = CovidenceAnnotationConverter()
            converter._infer_column_types(rows, fieldnames, sample_size=2)
            # Output: {"id": int, "age": int, "gender": str}

        """
        n_rows = len(rows)

        # Collect observed value types per column (skipping missing values).
        # Used as evidence for downstream column type inference.
        # Example: column_type_observations = {age: [int, int], gender: [str, str]}
        column_type_evidence: defaultdict[str, list] = defaultdict(list)

        if sample_size is None:
            sample_size = n_rows

        for col_name in column_names:
            row_num = 0

            while (
                len(column_type_evidence[col_name]) < sample_size and row_num < n_rows
            ):
                att_type = self._infer_type(rows[row_num][col_name])

                if att_type is not None:
                    column_type_evidence[col_name].append(att_type)

                row_num += 1

        resolved_column_types: dict[str, AttributeType] = {}
        for col_name, col_types in column_type_evidence.items():
            resolved_column_types[col_name] = self._resolve_types(col_types)

        return resolved_column_types

    @staticmethod
    def _build_destiny_reference_dict_from_row(
        mapping: dict[str, str],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a nested dictionary of reference data from a CSV row.

        Maps flat CSV columns to the nested structure expected by
        destiny_sdk.enhancements models, The resulting dictionary is used to create
        `EnhancementFileInput` and `ReferenceFileInput` objects.

        Args:
            mapping: Dictionary mapping nested keys(dot-separated strings) to CSV
            column names.
            row: Dictionary representing a CSV row.

        Returns:
            Nested dictionary representing the reference data. Example:

        -----------------------------------------------
        USER GUIDE
        -----------
        The `mapping` argument defines how CSV columns map to Destiny SDK fields.

        Format:
            mapping = {
                "<destiny_field>": "<csv_column_name>",
            }
        Rules:
            - Keys are dot-separated Destiny SDK field paths.
            - Values are CSV column names.

        SUPPORTED FIELDS
        ----------------
        - abstract
        - authorship
        - cited_by_count
        - created_date
        - updated_date
        - publication_date
        - publication_year
        - publisher
        - title
        - pagination.volume
        - pagination.issue
        - pagination.first_page
        - pagination.last_page
        - publication_venue.display_name
        - publication_venue.issn
        - publication_venue.venue_type
        - publication_venue.issn_l
        - publication_venue.host_organization_name

        -----------------------------------------------

        Example:
        mapping = {"publication_venue.display_name": "journal", "abstract": "abstract"}
        row = {"journal": "Nature Climate Change", "abstract": "ABCDEFXXXXXX"}
        # Output
            {
            "publication_venue": {"display_name": "Nature Climate Change"},
            "abstract": "ABCDEFXXXXXX"
            }


        """
        result: dict[str, Any] = {}

        for key, column_name in mapping.items():
            value = row[column_name].strip()
            if value == "":
                value = None

            parts = key.split(".")
            d = result

            for part in parts[:-1]:
                d = d.setdefault(part, {})

            d[parts[-1]] = value

        return result

    # TODO: Allow users to provide seperator
    def _build_destiny_authorship_list(self, authors: str) -> list[Authorship]:
        """
        Build a list of a `Authorship` objects  (as defined in destiny_sdk.enhancements)
        from a string. The returned list is in the format expected by
        `BibliographicMetadataEnhancement` for authorship.

        Args:
            authors: A semicolon-separated string of author names. eg:("Alice; Bob; Mo")

        Returns:
            List of `Authorship` objects representing each author with their position.

        """
        # Split on semicolons, strip whitespace, and remove any empty entries
        sep: str = self.config.author_separator
        clean_authors = [
            author.strip() for author in authors.split(sep) if author.strip()
        ]

        authorship: list[Authorship] = []
        if not clean_authors:
            return authorship

        for i, author in enumerate(clean_authors):
            position = AuthorPosition.MIDDLE
            if i == 0:
                position = AuthorPosition.FIRST
            elif i == len(clean_authors) - 1:
                position = AuthorPosition.LAST
            authorship.append(Authorship(display_name=author, position=position))
        return authorship

    def build_destiny_reference(
        self,
        row: dict,
        mapping: dict[str, Any],
        source: str = "deet CSV converter",
    ) -> ReferenceFileInput:
        """

        Convert a CSV row into a `ReferenceFileInput` for destiny_sdk.reference.

        This method extracts bibliographic metadata and abstract content from the
        given row using the provided reference field mapping. It constructs
        `BibliographicMetadataEnhancement` and `AbstractContentEnhancement`
        objects, then wraps them in `EnhancementFileInput` objects, and finally
        returns a `ReferenceFileInput`.

        Args:
            row: Dictionary representing a single CSV row.
            mapping: Dictionary mapping nested keys (dot-separated strings) to CSV
                 column names.
            source: Optional string indicating the source of the data; defaults to
                "deet CSV converter"
        Returns:
            ReferenceFileInput object containing all extracted enhancements

        """
        reference_dict = self._build_destiny_reference_dict_from_row(mapping, row)

        abstract = (
            AbstractContentEnhancement(
                abstract=reference_dict["abstract"],
                process=AbstractProcessType.UNINVERTED,
            )
            if "abstract" in reference_dict
            else None
        )

        reference_dict["authorship"] = (
            self._build_destiny_authorship_list(reference_dict["authorship"])
            if "authorship" in reference_dict
            and isinstance(reference_dict["authorship"], str)
            else None
        )

        bibliographic_data = BibliographicMetadataEnhancement(**reference_dict)

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
        file_path: Path,
        attribute_fields: list[str] | None = None,
        reference_fields: dict | None = None,
    ) -> tuple[list[str], dict[str, str], list[str], list[dict[str, Any]]]:
        """
        Load a CSV, normalize headers, and return all column names, attribute names,
        reference names, and rows.
        """
        path = Path(file_path)
        with path.open(newline="", encoding="utf-8-sig") as f:
            csv_reader = csv.DictReader(f)

            # normalize headers BEFORE reading rows
            raw_headers = csv_reader.fieldnames or []
            colnames: list[str] = [h.strip().lower() for h in raw_headers]
            csv_reader.fieldnames = colnames

            # --- validate duplicates ---
            dup_fields = self._find_duplicate_column_names(colnames)
            if dup_fields:
                msg = f"{len(dup_fields)} Duplicate fieldnames found: {dup_fields}"
                raise ValueError(msg)

            # --- validate required fields ---
            meta_fields = {"name", "document_id"}
            missing = meta_fields - set(colnames)
            if missing:
                msg = f"Required columns missing: {missing}"
                raise ValueError(msg)

            # --- validate reference fields ---
            # If reference_fields is not povided....
            if reference_fields is None:
                if self.config.auto_assign_reference_fields:
                    logger.info("Auto assigning reference fields")
                    reference_fields = {
                        ref_field: ref_field
                        for ref_field in ALLOWED_REFERENCE_MAPPING_KEYS
                        if ref_field in colnames
                    }
                else:
                    logger.info("No reference fields provided and auto assign is False")
                    reference_fields = {}

            # If reference_fields is povided....
            if reference_fields:
                invalid_keys = set(reference_fields) - ALLOWED_REFERENCE_MAPPING_KEYS
                if invalid_keys:
                    msg = f"Invalid mapping keys: {invalid_keys}"
                    raise ValueError(msg)

                # normalize reference fields
                reference_fields = {
                    k: v.strip().lower() for k, v in reference_fields.items()
                }

                unknown = set(reference_fields.values()) - set(colnames)
                if unknown:
                    msg = f"Reference fields not found in CSV: {unknown}"
                    raise ValueError(msg)

            # --- validate attribute fields ---
            # normalize and validate provided attribute fields
            if attribute_fields is None:
                logger.info("No attribute fields provided")
                excluded_fields = meta_fields | set(reference_fields.values())
                resolved_attribute_fields = [
                    h for h in colnames if h not in excluded_fields
                ]
            else:
                resolved_attribute_fields = [
                    field.strip().lower() for field in attribute_fields
                ]
                unknown_attributes = set(resolved_attribute_fields) - set(colnames)
                if unknown_attributes:
                    msg = f"Attribute fields not found in CSV: {unknown_attributes}"
                    raise ValueError(msg)

            # Read rows into mem once
            rows = list(csv_reader)

        return colnames, reference_fields, resolved_attribute_fields, rows

    def build_attributes(
        self,
        attribute_fields: list[str],
        rows: list[dict],
    ) -> list[Attribute]:
        """
        Build a list of `Attribute` objects from CSV rows and specified attribute
        columns.

        Infers the `AttributeType` for each attribute field and assigns a unique ID
        to each `Attribute`.
        """
        attribute_types = self._infer_column_types(rows, attribute_fields)
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
        reference_fields: dict,
        rows: list[dict],
    ) -> tuple[list[Document], list[GoldStandardAnnotatedDocument]]:
        """
        Build `Document` objects and their corresponding GoldStandardAnnotatedDocument`s
        from CSV rows.

        Args:
            attributes:
            reference_fields: Dictionary mapping reference field labels (as defined in
                destiny_sdk.enhancements) to CSV column names.

            rows: List of dictionaries representing all CSV rows.

        Returns:
            A tuple containing: Document and List[GoldStandardAnnotatedDocument]

        """
        annotated_documents: list[GoldStandardAnnotatedDocument] = []
        documents: list[Document] = []

        # --- To acces attributes by their names ---
        attr_by_label = {a.attribute_label: a for a in attributes}

        for row in rows:
            # --- Build destiny Reference given row ---
            row_reference = self.build_destiny_reference(row, reference_fields)
            # --- Build Document ---
            document = Document(
                name=row["name"],
                citation=row_reference,
                document_id=row["document_id"],
            )
            document.init_document_identity()
            documents.append(document)

            # --- Build Document Annotations ---
            annotations = []
            for label, attr in attr_by_label.items():
                raw_value = row[label].strip()

                annotation = GoldStandardAnnotation(
                    attribute=attr,
                    raw_data=raw_value,
                    annotation_type=AnnotationType.HUMAN,
                )

                annotations.append(annotation)

            # --- Build Annotated Documents = Attach annotations to document ---
            annotated_doc = GoldStandardAnnotatedDocument(
                document=document,
                annotations=annotations,
            )

            annotated_documents.append(annotated_doc)

        return documents, annotated_documents

    def process_annotation_file(
        self,
        file_path: str | Path,
        set_attribute_type: str | AttributeType | None = None,
        attribute_fields: list | None = None,
        reference_fields: dict | None = None,
    ) -> ProcessedAnnotationData:
        """
        Process a complete CSV annotation file and return structured data.

        Each row is assumed to represent a document, and columns correspond to
        different types of fields (metadata, reference information, and attributes).

        Args:
            file_path: Path to the CSV annotation file.
            attribute_fields: List of column names to be treated as document attributes.
            reference_fields: Dictionary mapping reference field labels(as defined in
            destiny_sdk.enhancements) to corresponding CSV column names.


        Returns:
            ProcessedAnnotationData containing all processed data.

        """
        file_path = Path(file_path)

        if set_attribute_type is not None:
            msg = (
                "CsvAnnotationConverter does not support set_attribute_type; "
                "use attribute_fields instead."
            )
            raise NotImplementedError(msg)
        logger.info(f"Processing annotation file: {file_path}")

        colnames, reference_fields, attribute_fields, rows = self.load_csv(
            file_path, attribute_fields, reference_fields
        )

        attributes = self.build_attributes(attribute_fields, rows)

        documents, annotated_documents = self.build_documents_and_annotations(
            attributes, reference_fields, rows
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
            annotations=[],  # keep type compatible
        )
