"""Convert annotation JSON files to Pydantic models."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import TypeAdapter

from deet.data_models.base import AnnotationType, AttributeType
from deet.data_models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
    EppiRawData,
)
from deet.data_models.processed_gold_standard_annotations import (
    ProcessedEppiAnnotationData,
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


class EppiAnnotationConverter(AnnotationConverter):
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
        Initialise the converter with configurable output paths.
        Set self.OUTFILE_LOADERS mapping the outfiles to be read/written to a
        filename, and a TypeAdapter defining the type of Pydantic Model to
        read back in when deserialising.

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
        self.OUTFILE_LOADERS: dict[Outfiles, tuple[str, TypeAdapter]] = {
            Outfiles.ATTRIBUTES: (
                attributes_filename,
                TypeAdapter(list[EppiAttribute]),
            ),
            Outfiles.DOCUMENTS: (
                documents_filename,
                TypeAdapter(list[EppiDocument]),
            ),
            Outfiles.ANNOTATED_DOCUMENTS: (
                annotated_documents_filename,
                TypeAdapter(list[EppiGoldStandardAnnotatedDocument]),
            ),
            Outfiles.ATTRIBUTE_LABEL_MAPPING: (
                attribute_mapping_filename,
                TypeAdapter(dict[int, str]),
            ),
        }

    @property
    def processed_data_type(self) -> type[ProcessedEppiAnnotationData]:
        """Return ProcessedEppiAnnotationData."""
        return ProcessedEppiAnnotationData

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

    def _extract_attributes_from_codesets(
        self, raw_data: EppiRawData
    ) -> list[dict[str, Any]]:
        """Extract and flatten attributes from CodeSets using structured models."""
        return raw_data.extract_all_attributes(self.flatten_attributes_hierarchy)

    def convert_to_eppi_attributes(
        self,
        flattened_attributes: list[dict[str, Any]],
        set_attribute_type: AttributeType | None = None,
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
            new_attribute = EppiAttribute(**att_dict)
            if set_attribute_type:
                logger.debug(
                    f"setting custom attribute type {set_attribute_type.value} "
                    f"for attribute {new_attribute.attribute_id}"
                )
                new_attribute.output_data_type = set_attribute_type
            out.append(new_attribute)
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

        output_data: str | bool | int | float | list | dict = (
            " | ".join(extracted_texts) if extracted_texts else ""
        )

        # Coerce output_data to match attribute's output_data_type. Required because
        # EppiGoldStandardAnnotation validates type. When ItemAttributeFullTextDetails
        # is absent, use type-appropriate defaults. For BOOL with text, use truthiness.
        attr = attributes_lookup.get(annotation.get("AttributeId", 0))
        if attr is not None:
            if not extracted_texts:
                output_data = attr.output_data_type.to_python_type()()
            elif attr.output_data_type == AttributeType.BOOL:
                output_data = bool(output_data)

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

    def process_annotation_file(
        self,
        file_path: str | Path,
        set_attribute_type: AttributeType | str | None = None,
    ) -> ProcessedEppiAnnotationData:
        """
        Process a complete annotation file and return structured data.

        Args:
            file_path: Path to the JSON annotation file
            set_attribute_type: custom AttributeType to set for incoming annotations.

        Returns:
            ProcessedAnnotationData containing all processed data

        """
        logger.info(f"Processing annotation file: {file_path}")

        with Path(file_path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_data = EppiRawData.model_validate(data)

        all_attributes_raw = self._extract_attributes_from_codesets(raw_data)

        if isinstance(set_attribute_type, str):
            set_attribute_type = AttributeType(set_attribute_type)
        attributes = self.convert_to_eppi_attributes(
            flattened_attributes=all_attributes_raw,
            set_attribute_type=set_attribute_type,
        )

        attributes_lookup: dict[int, EppiAttribute] = {
            attr.attribute_id: attr for attr in attributes
        }

        attribute_id_to_label: dict[int, str] = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        annotated_documents = []
        all_annotations = []
        documents_by_item_id: dict[int, EppiDocument] = {}

        # Process each reference with its annotations directly
        # Annotations are already nested within their parent reference
        for reference in data.get("References", []):
            item_id = reference.get("ItemId")
            if item_id is None:
                continue

            # Create or retrieve document using ItemId as unique identifier
            if item_id not in documents_by_item_id:
                document = EppiDocument.model_validate(reference)
                documents_by_item_id[item_id] = document
            else:
                document = documents_by_item_id[item_id]

            # Get annotations directly from this reference's Codes array
            reference_codes = reference.get("Codes", [])
            if reference_codes:
                annotations = self.convert_to_eppi_annotations(
                    reference_codes,
                    attributes_lookup,
                    attribute_id_to_label,
                )
            else:
                annotations = []

            annotated_doc = EppiGoldStandardAnnotatedDocument(
                document=document, annotations=annotations
            )

            annotated_documents.append(annotated_doc)
            all_annotations.extend(annotations)

        logger.info(
            f"Processed {len(attributes)} attributes,"
            f" {len(documents_by_item_id)} documents, "
            f"{len(all_annotations)} annotations,"
            f" {len(annotated_documents)} annotated documents"
        )

        return ProcessedEppiAnnotationData(
            attributes=attributes,
            documents=list(documents_by_item_id.values()),
            annotations=all_annotations,
            annotated_documents=annotated_documents,
            attribute_id_to_label=attribute_id_to_label,
            raw_data=raw_data,
        )
