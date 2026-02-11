"""Convert annotation JSON files to Pydantic models."""

import json
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

from loguru import logger

from deet.data_models.base import AnnotationType, AttributeType
from deet.data_models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
    EppiRawData,
    ProcessedAnnotationData,
)

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


class EppiAnnotationConverter:
    """
    A class to convert raw EPPI-Reviewer JSON annotations
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
            Outfiles.ATTRIBUTE_LABEL_MAPPING: attribute_mapping_filename,
        }

    def flatten_attributes_hierarchy(
        self, attributes_list: list[dict[str, Any]], parent_path: str = ""
    ) -> list[dict[str, Any]]:
        """
        Recursively flatten the hierarchical attributes structure.

        Args:
            attributes_list: List of attribute dictionaries from the JSON
            parent_path: Path to the parent attribute (for hierarchy tracking)

        Returns:
            List of flattened attribute dictionaries with hierarchy information

        """
        flattened = []

        for attr in attributes_list:
            # extract children before modifying  dict
            child_attributes = attr.get("Attributes", {}).get("AttributesList", [])

            attr["hierarchy_path"] = parent_path
            attr["hierarchy_level"] = (
                len(parent_path.split(" > ")) if parent_path else 0
            )
            attr["is_leaf"] = not bool(child_attributes)

            flattened.append(attr)

            # recursive extension
            if child_attributes:
                current_path = (
                    f"{parent_path} > {attr.get('AttributeName', '')}"
                    if parent_path
                    else attr.get("AttributeName", "")
                )
                flattened.extend(
                    self.flatten_attributes_hierarchy(child_attributes, current_path)
                )

        return flattened

    def convert_to_eppi_attributes(
        self, flattened_attributes: list[dict[str, Any]]
    ) -> list[EppiAttribute]:
        """
        Convert flattened attribute data to EppiAttribute models.

        Args:
            flattened_attributes: List of flattened attribute dictionaries

        Returns:
            List of EppiAttribute models

        """
        out = []
        for att_dict in flattened_attributes:
            if "AttributeId" not in att_dict:
                att_dict["AttributeId"] = 0
            out.append(EppiAttribute(**att_dict))
        return out

    def _process_text_details(
        self, text_details: list[dict[str, Any]]
    ) -> tuple[list[str], list[EppiItemAttributeFullTextDetails]]:
        """
        Process ItemAttributeFullTextDetails to extract texts and create detail objects.

        Args:
            text_details: List of text detail dictionaries from EPPI JSON

        Returns:
            Tuple of (extracted_texts, item_attribute_details)

        """
        extracted_texts = []
        item_attribute_details = []

        for text_detail in text_details:
            text = text_detail.get("Text", "")
            if text:
                extracted_texts.append(text)

            detail = EppiItemAttributeFullTextDetails(
                item_document_id=text_detail.get("ItemDocumentId"),
                text=text,
                item_arm=text_detail.get("ItemArm", ""),
            )
            item_attribute_details.append(detail)

        return extracted_texts, item_attribute_details

    def _convert_single_annotation(
        self,
        annotation: dict[str, Any],
        attributes_lookup: dict[int, EppiAttribute],
        attribute_id_to_label: dict[int, str] | None = None,
    ) -> EppiGoldStandardAnnotation:
        """
        Convert a single annotation dictionary to EppiGoldStandardAnnotation.

        Args:
            annotation: Single annotation dictionary from EPPI JSON
            attributes_lookup: Lookup dictionary for attributes
            attribute_id_to_label: Mapping from attribute ID to label

        Returns:
            EppiGoldStandardAnnotation model

        Note:
            If attribute is not found in lookup, creates a basic attribute using
            the attribute_id_to_label mapping. All annotations from JSON are
            marked as HUMAN type. Output data is converted to boolean (EPPI's
            output data type).

        """
        text_details = annotation.get("ItemAttributeFullTextDetails", [])
        extracted_texts, item_attribute_details = self._process_text_details(
            text_details
        )

        output_data: str | bool = " | ".join(extracted_texts) if extracted_texts else ""

        # Coerce empty string to False for BOOL attributes (backward compatibility
        # when ItemAttributeFullTextDetails is absent)
        attr = attributes_lookup.get(annotation.get("AttributeId", 0))
        if (
            attr is not None
            and attr.output_data_type == AttributeType.BOOL
            and output_data == ""
        ):
            output_data = False

        # Look up the attribute from the attributes list
        if (attribute_id := annotation.get("AttributeId")) is None:
            missing_attr_id_msg = (
                "Annotation is missing required field 'AttributeId'. "
                "All annotations must have an AttributeId."
            )
            raise ValueError(missing_attr_id_msg)

        # find attribute in attributes_lookup
        if (attribute := attributes_lookup.get(attribute_id)) is None:
            attr_not_found_msg = (
                f"Attribute with ID {attribute_id} not found in attributes list. "
                "All annotations must reference a valid attribute from the CodeSets."
            )
            raise ValueError(attr_not_found_msg)

        # ensure the attribute has the correct label from the mapping if available
        if attribute_id_to_label is not None and attribute_id in attribute_id_to_label:
            attribute.attribute_label = attribute_id_to_label[attribute_id]

        return EppiGoldStandardAnnotation(
            attribute=attribute,
            additional_text=annotation.get("AdditionalText", ""),
            arm_id=annotation.get("ArmId"),
            arm_title=annotation.get("ArmTitle", ""),
            arm_description=annotation.get("ArmDescription", ""),
            output_data=output_data,
            annotation_type=AnnotationType.HUMAN,
            item_attribute_full_text_details=item_attribute_details,
        )

    def convert_to_eppi_annotations(
        self,
        annotations_data: list[dict[str, Any]],
        attributes_lookup: dict[int, EppiAttribute],
        attribute_id_to_label: dict[int, str] | None = None,
    ) -> list[EppiGoldStandardAnnotation]:
        """
        Convert several dicts to a list of EppiGoldStandardAnnotations.

        Args:
            annotations_data: List of human, gold standard
                annotation dicts from EPPI JSON
            document: The document these annotations belong to
            attributes_lookup: Lookup dictionary for attributes
            attribute_id_to_label: Mapping from attribute ID to label

        Returns:
            List of EppiGoldStandardAnnotation models

        """
        return [
            self._convert_single_annotation(
                annotation, attributes_lookup, attribute_id_to_label
            )
            for annotation in annotations_data
        ]

    def _extract_attributes_from_codesets(
        self, raw_data: EppiRawData
    ) -> list[dict[str, Any]]:
        """Extract and flatten attributes from CodeSets using structured models."""
        return raw_data.extract_all_attributes(self.flatten_attributes_hierarchy)

    def _create_pdf_to_title_mapping(
        self, references: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Create mapping from PDF filenames to document titles."""
        pdf_to_title_mapping = {}
        for ref in references:
            title = ref.get("Title", "")
            pdf_filename = title.replace(" ", "_") + ".pdf"
            pdf_to_title_mapping[pdf_filename] = title

            year = ref.get("Year", "")
            if year:
                authors = (
                    ref.get("Authors", "").split(";")[0].strip()
                    if ref.get("Authors")
                    else ""
                )
                if authors:
                    author_year_pdf = f"{authors.split()[0]} {year}.pdf"
                    pdf_to_title_mapping[author_year_pdf] = title

        return pdf_to_title_mapping

    def _find_document_annotations(
        self,
        all_annotations_raw: list[dict[str, Any]],
        doc_title: str,
        pdf_to_title_mapping: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Find annotations for a specific document."""
        doc_annotations = []
        for ann in all_annotations_raw:
            for text_detail in ann.get("ItemAttributeFullTextDetails", []):
                doc_title_from_ann = text_detail.get("DocTitle", "")
                if doc_title_from_ann == doc_title or (
                    doc_title_from_ann in pdf_to_title_mapping
                    and pdf_to_title_mapping[doc_title_from_ann] == doc_title
                ):
                    doc_annotations.append(ann)
                    break

        return doc_annotations

    def process_annotation_file(self, file_path: str | Path) -> ProcessedAnnotationData:
        """
        Process a complete annotation file and return structured data.

        Args:
            file_path: Path to the JSON annotation file

        Returns:
            ProcessedAnnotationData containing all processed data

        """
        logger.info(f"Processing annotation file: {file_path}")

        with Path(file_path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_data = EppiRawData.model_validate(data)

        all_attributes_raw = self._extract_attributes_from_codesets(raw_data)

        attributes = self.convert_to_eppi_attributes(all_attributes_raw)

        attributes_lookup: dict[int, EppiAttribute] = {
            attr.attribute_id: attr for attr in attributes
        }

        attribute_id_to_label: dict[int, str] = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        all_annotations_raw = []
        documents_by_title: dict[str, EppiDocument] = {}

        for reference in data.get("References", []):
            reference_codes = reference.get("Codes", [])
            all_annotations_raw.extend(reference_codes)

            doc_title = reference.get("Title", "")
            if doc_title and doc_title not in documents_by_title:
                document = EppiDocument(**reference)
                documents_by_title[doc_title] = document

        pdf_to_title_mapping = self._create_pdf_to_title_mapping(
            data.get("References", [])
        )

        annotated_documents = []
        all_annotations = []

        for doc_title, doc in documents_by_title.items():
            doc_annotations = self._find_document_annotations(
                all_annotations_raw, doc_title, pdf_to_title_mapping
            )

            if doc_annotations:
                annotations = self.convert_to_eppi_annotations(
                    doc_annotations,
                    attributes_lookup,
                    attribute_id_to_label,
                )
                payload = doc.model_dump(mode="python")
                annotations = [json.loads(ann.model_dump_json()) for ann in annotations]

                payload["annotations"] = annotations

                # annotated_doc = EppiGoldStandardAnnotatedDocument(**payload)
                annotated_doc = EppiGoldStandardAnnotatedDocument(**payload)

                annotated_documents.append(annotated_doc)
                all_annotations.extend(annotations)

        logger.info(
            f"Processed {len(attributes)} attributes,"
            " {len(documents_by_title)} documents, "
            f"{len(all_annotations)} annotations,"
            " {len(annotated_documents)} annotated documents"
        )

        return ProcessedAnnotationData(
            attributes=attributes,
            documents=list(documents_by_title.values()),
            annotations=all_annotations,
            annotated_documents=annotated_documents,
            attribute_id_to_label=attribute_id_to_label,
            raw_data=raw_data,
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
