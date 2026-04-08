"""Convert annotation CSV files to Pydantic models."""

import csv
from collections import defaultdict
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
from pydantic import TypeAdapter

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
from deet.logger import logger
from deet.processors.base_converter import (
    DEFAULT_ANNOTATED_DOCUMENTS_FILENAME,
    DEFAULT_ATTRIBUTE_MAPPING_FILENAME,
    DEFAULT_ATTRIBUTES_FILENAME,
    DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_DOCUMENTS_FILENAME,
    AnnotationConverter,
    Outfiles,
)


class CSVAnnotationConverter(AnnotationConverter):
    """
    A class to convert raw CSV (e.g. Covidence) annotations
    into structured Pydantic models.

     This converter operates on flat CSV columns, infers field/column types,
    and produces attributes, documents, and annotated document records.
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
            base_output_dir: Base directory for saving processed files
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
    def _normalize_text(value: str) -> str:
        """Strip white space and lowercase text."""
        return value.strip().lower()

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

    # TODO: need to add list and dictionary handling
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
        msg = "Multiple incompatible types found"
        raise ValueError(msg)

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
        raw_column_types: defaultdict[str, list] = defaultdict(list)

        if sample_size is None:
            sample_size = n_rows

        for col_name in column_names:
            row_num = 0

            while len(raw_column_types[col_name]) < sample_size and row_num < n_rows:
                att_type = self._infer_type(rows[row_num][col_name])

                if att_type is not None:
                    raw_column_types[col_name].append(att_type)

                row_num += 1

        resolved_column_types: dict[str, AttributeType] = {}
        for col_name, col_types in raw_column_types.items():
            try:
                resolved_column_types[col_name] = self._resolve_types(col_types)
            except ValueError as e:
                msg = f"Error inferring type for field '{col_name}': {e}"
                raise ValueError(msg) from e

        return resolved_column_types

    @staticmethod
    def _build_reference_dict_from_row(
        mapping: dict[str, str],
        row: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a nested dictionary of reference data from a CSV row.

        Maps flat CSV columns to the nested structure expected by
        destiny_sdk.enhancements models, The resulting dictionary can be used to create
        `EnhancementFileInput` and `ReferenceFileInput` objects.

        Args:
            mapping: Dictionary mapping nested keys(dot-separated strings) to CSV
            column names.
            row: Dictionary representing a CSV row.

        Returns:
            Nested dictionary representing the reference data. Example:

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
            if column_name not in row:
                continue  # or raise, depending on strictness

            parts = key.split(".")
            d = result

            for part in parts[:-1]:
                d = d.setdefault(part, {})

            d[parts[-1]] = row[column_name].strip()

        return result

    @staticmethod
    def _build_authorship_list(authors: str) -> list[Authorship]:
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
        clean_authors = [
            author.strip() for author in authors.split(";") if author.strip()
        ]

        authorship: list[Authorship] = []
        if not clean_authors:
            return authorship

        for i, author in enumerate(clean_authors):
            position = AuthorPosition.MIDDLE
            if i == 0:
                position = AuthorPosition.FIRST
            if i == len(clean_authors) - 1:
                position = AuthorPosition.LAST
            authorship.append(Authorship(display_name=author, position=position))
        return authorship

    def build_destiny_reference(
        self,
        row: dict,
        mapping: dict[str, Any],
        source: str = "DEET CSV converter",
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
                "DEET CSV converter"
        Returns:
            ReferenceFileInput object containing all extracted enhancements

        """
        reference_dict = self._build_reference_dict_from_row(mapping, row)

        abstract = (
            AbstractContentEnhancement(
                abstract=reference_dict["abstract"],
                process=AbstractProcessType.UNINVERTED,
            )
            if "abstract" in reference_dict
            else None
        )

        reference_dict["authorship"] = (
            self._build_authorship_list(reference_dict["authorship"])
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
        file_path: str | Path,
        attribute_fields: list[str] | None = None,
        reference_fields: dict | None = None,
    ) -> tuple[list[str], dict[str, str], list[str], list[dict[str, Any]]]:
        """
        Load a CSV, normalize headers, and return all column names, attribute names,
        reference names, and rows.
        """
        path = Path(file_path)
        with path.open(newline="") as f:
            csv_reader = csv.DictReader(f)

            # normalize headers BEFORE reading rows
            raw_headers = csv_reader.fieldnames or []
            colnames: list[str] = [self._normalize_text(h) for h in raw_headers]
            csv_reader.fieldnames = colnames

            # validate duplicates
            dup_fields = self._find_duplicate_column_names(colnames)
            if dup_fields:
                msg = f"{len(dup_fields)} Duplicate fieldnames found: {dup_fields}"
                raise ValueError(msg)

            # validate required fields
            meta_fields = {"name", "document_id"}
            missing = meta_fields - set(colnames)
            if missing:
                msg = f"Required columns missing: {missing}"
                raise ValueError(msg)

            if reference_fields is None:
                logger.info("No reference fields provided")
                reference_fields = {}
            else:
                # normalize reference fields
                reference_fields = {
                    k: self._normalize_text(v) for k, v in reference_fields.items()
                }

                unknown = set(reference_fields.values()) - set(colnames)
                if unknown:
                    msg = f"Reference fields not found in CSV: {unknown}"
                    raise ValueError(msg)

            # normalize and validate provided attribute fields
            if attribute_fields is None:
                logger.info("No attribute fields provided")
                excluded_fields = meta_fields | set(reference_fields.values())
                resolved_attribute_fields = [
                    h for h in colnames if h not in excluded_fields
                ]
            else:
                resolved_attribute_fields = [
                    self._normalize_text(field) for field in attribute_fields
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
        attr_by_label = {a.attribute_label: a for a in attributes}

        # ---  ---
        annotated_documents: list[GoldStandardAnnotatedDocument] = []
        documents: list[Document] = []

        for row_idx, row in enumerate(rows):
            row_reference = self.build_destiny_reference(row, reference_fields)

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
