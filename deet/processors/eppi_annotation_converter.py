"""Convert annotation JSON files to Pydantic models."""

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from destiny_sdk.enhancements import Visibility
from destiny_sdk.references import Reference

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
from deet.logger import logger


class EppiAnnotationConverter:
    """
    A class to convert raw EPPI-Reviewer JSON annotations
    into structured Pydantic models.

    This converter handles the complex hierarchical
    structure of EPPI attributes by flattening
    them while preserving parent-child relationships
    through path information.
    """

    def process_attribute_data_for_validation(
        self, attr_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process raw attribute data for EppiAttribute validation.

        Maps camelCase JSON fields to snake_case Python fields and handles
        fields that need manual processing. Explicitly maps camelCase to snake_case
        since alias generators don't work in reverse for deserialization.

        For EPPI attributes: question_target is always empty, output_data_type
        is always boolean. Hierarchy fields (hierarchy_path, hierarchy_level, is_leaf)
        are already in snake_case from flatten_attributes_hierarchy.

        Args:
            attr_data: Raw attribute data from EPPI JSON

        Returns:
            Dictionary with fields mapped to snake_case Python field names

        """
        return {
            # Core fields that need manual processing
            "question_target": "",  # Always empty for EPPI
            "output_data_type": AttributeType.BOOL,  # Always boolean for EPPI
            "attribute_id": attr_data.get("AttributeId", 0),  # Keep as int
            "attribute_label": attr_data.get("AttributeName", ""),
            "attribute_description": attr_data.get("AttributeDescription"),
            "attribute_type": attr_data.get("AttributeType"),
            "attribute_set_description": attr_data.get("AttributeSetDescription"),
        }

    def process_document_data_for_validation(
        self, document_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process raw document data for EppiDocument validation.

        Handles fields that need manual processing (name from Title, context from
        Abstract, document_id from ItemId, citation object creation).
        All EPPI-specific fields (item_id, title, parent_title, etc.) are
        automatically mapped by alias generators from camelCase JSON.

        Args:
            document_data: Raw document data from EPPI JSON

        Returns:
            Dictionary with only the fields that need manual processing

        """
        return {
            "name": document_data.get("Title"),
            "citation": self._create_reference(document_data),
            "context": document_data.get("Abstract"),
            "document_id": str(document_data.get("ItemId", "")),
            "filename": document_data.get("Title", "").replace(" ", "_") + ".pdf"
            if document_data.get("Title")
            else None,
        }

    def _create_reference(self, document_data: dict[str, Any]) -> Reference:
        """Create a Reference object from document data."""
        return Reference(
            id=uuid4(),
            visibility=Visibility.PUBLIC,
        )

    def load_eppi_json_annotations(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load EPPI-Reviewer JSON annotations from a file.

        Args:
            file_path: Path to the EPPI JSON annotation file

        Returns:
            Dictionary containing the loaded EPPI annotations

        """
        with Path(file_path).open("r", encoding="utf-8") as f:
            return json.load(f)

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
            flattened_attr = {
                "AttributeId": attr.get("AttributeId"),
                "AttributeName": attr.get("AttributeName"),
                "AttributeDescription": attr.get("AttributeDescription"),
                "AttributeSetDescription": attr.get("AttributeSetDescription"),
                "AttributeType": attr.get("AttributeType"),
                "AttributeTypeId": attr.get("AttributeTypeId"),
                "AttributeSetId": attr.get("AttributeSetId"),
                "OriginalAttributeID": attr.get("OriginalAttributeID"),
                "ExtURL": attr.get("ExtURL"),
                "ExtType": attr.get("ExtType"),
                "hierarchy_path": parent_path,
                "hierarchy_level": len(parent_path.split(" > ")) if parent_path else 0,
                "is_leaf": "Attributes" not in attr
                or not attr["Attributes"].get("AttributesList"),
            }

            flattened.append(flattened_attr)

            if "Attributes" in attr and "AttributesList" in attr["Attributes"]:
                child_attributes = attr["Attributes"]["AttributesList"]
                current_path = (
                    f"{parent_path} > {attr.get('AttributeName', '')}"
                    if parent_path
                    else attr.get("AttributeName", "")
                )
                child_flattened = self.flatten_attributes_hierarchy(
                    child_attributes, current_path
                )
                flattened.extend(child_flattened)

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
        attributes = []

        for attr_data in flattened_attributes:
            manual_fields = self.process_attribute_data_for_validation(attr_data)

            combined_data = {
                **attr_data,
                **manual_fields,
            }

            logger.debug(combined_data)
            attribute = EppiAttribute.model_validate(combined_data)
            attributes.append(attribute)

        return attributes

    def convert_to_eppi_document(self, document_data: dict[str, Any]) -> EppiDocument:
        """
        Convert document data to EppiDocument model.

        Args:
            document_data: Dictionary containing document information

        Returns:
            EppiDocument model

        """
        manual_fields = self.process_document_data_for_validation(document_data)
        combined_data = {**document_data, **manual_fields}
        return EppiDocument.model_validate(combined_data)

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
        attributes_lookup: dict[int, EppiAttribute] | None = None,
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

        output_data = " | ".join(extracted_texts) if extracted_texts else ""

        # Look up the attribute from the attributes list
        #
        # NOTE: EPPI JSON uses int `AttributeId`. Our `EppiAttribute.attribute_id`
        # is also an int. Python's json parser keeps numeric values as int, so
        # we can directly use the value without conversion.
        attribute_id = annotation.get("AttributeId")

        # Validate that attribute_id is present (required field)
        if attribute_id is None:
            missing_attr_id_msg = (
                "Annotation is missing required field 'AttributeId'. "
                "All annotations must have an AttributeId."
            )
            raise ValueError(missing_attr_id_msg)

        # Validate that attributes_lookup is provided
        if attributes_lookup is None:
            missing_lookup_msg = (
                "attributes_lookup is required but was None. "
                "Cannot convert annotation without attribute definitions."
            )
            raise ValueError(missing_lookup_msg)

        # Look up the attribute - it must exist in the attributes list
        attribute = attributes_lookup.get(attribute_id)

        if attribute is None:
            attr_not_found_msg = (
                f"Attribute with ID {attribute_id} not found in attributes list. "
                "All annotations must reference a valid attribute from the CodeSets."
            )
            raise ValueError(attr_not_found_msg)

        # Ensure the attribute has the correct label from the mapping if available
        if attribute_id_to_label is not None and attribute_id in attribute_id_to_label:
            attribute.attribute_label = attribute_id_to_label[attribute_id]

        return EppiGoldStandardAnnotation(
            attribute=attribute,
            additional_text=annotation.get("AdditionalText", ""),
            arm_id=annotation.get("ArmId"),
            arm_title=annotation.get("ArmTitle", ""),
            arm_description=annotation.get("ArmDescription", ""),
            output_data=bool(output_data),
            annotation_type=AnnotationType.HUMAN,
            item_attribute_full_text_details=item_attribute_details,
        )

    def convert_to_eppi_annotations(
        self,
        annotations_data: list[dict[str, Any]],
        document: EppiDocument,
        attributes_lookup: dict[int, EppiAttribute] | None = None,
        attribute_id_to_label: dict[int, str] | None = None,
    ) -> list[EppiGoldStandardAnnotation]:
        """
        Convert annotation data to EppiGoldStandardAnnotation models.

        Args:
            annotations_data: List of annotation objects from EPPI JSON
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

    def process_annotation_file(self, file_path: str | Path) -> ProcessedAnnotationData:
        """
        Process a complete annotation file and return structured data.

        Args:
            file_path: Path to the JSON annotation file

        Returns:
            ProcessedAnnotationData containing all processed data

        """
        logger.info(f"Processing annotation file: {file_path}")

        data = self.load_eppi_json_annotations(file_path)
        raw_data = EppiRawData.model_validate(data)

        all_attributes_raw = self._extract_attributes_from_codesets(raw_data)

        attributes = self.convert_to_eppi_attributes(all_attributes_raw)

        attributes_lookup: dict[int, EppiAttribute] = {
            attr.attribute_id: attr for attr in attributes
        }

        attribute_id_to_label: dict[int, str] = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        documents_by_item_id: dict[int, EppiDocument] = {}
        annotated_documents = []
        all_annotations = []

        # Process each reference with its annotations directly
        # Annotations are already nested within their parent reference
        for reference in data.get("References", []):
            item_id = reference.get("ItemId")
            if item_id is None:
                continue

            # Create or retrieve document using ItemId as unique identifier
            if item_id not in documents_by_item_id:
                document = self.convert_to_eppi_document(reference)
                documents_by_item_id[item_id] = document
            else:
                document = documents_by_item_id[item_id]

            # Get annotations directly from this reference's Codes array
            reference_codes = reference.get("Codes", [])
            if reference_codes:
                annotations = self.convert_to_eppi_annotations(
                    reference_codes,
                    document,
                    attributes_lookup,
                    attribute_id_to_label,
                )

                annotated_doc = EppiGoldStandardAnnotatedDocument(
                    **document.model_dump(), annotations=annotations
                )
                annotated_documents.append(annotated_doc)
                all_annotations.extend(annotations)

        logger.info(
            f"Processed {len(attributes)} attributes,"
            f" {len(documents_by_item_id)} documents, "
            f"{len(all_annotations)} annotations,"
            " {len(annotated_documents)} annotated documents"
        )

        return ProcessedAnnotationData(
            attributes=attributes,
            documents=list(documents_by_item_id.values()),
            annotations=all_annotations,
            annotated_documents=annotated_documents,
            attribute_id_to_label=attribute_id_to_label,
            raw_data=raw_data,
        )

    def save_processed_data(
        self,
        processed_data: ProcessedAnnotationData,
        output_dir: str | Path,
        input_filename: str | None = None,
    ) -> dict[str, str]:
        """
        Save processed data to structured files using Pydantic model serialization.

        Args:
            processed_data: The processed data from process_annotation_file
            output_dir: Directory to save the processed files
                        (required - no default to prevent accidental commits)
            input_filename: Optional filename to create a subdirectory
                            (if not provided, saves directly to output_dir)

        Returns:
            Dictionary mapping data types to saved file paths

        """
        base_path = Path(output_dir)
        eppi_base_path = base_path

        if input_filename:
            filename_without_ext = Path(input_filename).stem
            eppi_path = eppi_base_path / filename_without_ext
        else:
            eppi_path = eppi_base_path

        eppi_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        file_mappings = {
            "attributes": processed_data.attributes,
            "documents": processed_data.documents,
            "annotated_documents": processed_data.annotated_documents,
        }

        for file_type, data_list in file_mappings.items():
            file_path = eppi_path / f"{file_type}.json"
            file_path.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in data_list],  # type: ignore[attr-defined]
                    indent=2,
                )
            )
            saved_files[file_type] = str(file_path)

        mapping_file = eppi_path / "attribute_id_to_label_mapping.json"
        mapping_file.write_text(
            json.dumps(processed_data.attribute_id_to_label, indent=2)
        )
        saved_files["attribute_mapping"] = str(mapping_file)

        return saved_files
