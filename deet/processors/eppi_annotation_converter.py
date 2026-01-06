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

        Maps camelCase JSON fields to snake_case Python fields
        and handles fields that need manual processing.

        Args:
            attr_data: Raw attribute data from EPPI JSON

        Returns:
            Dictionary with fields mapped to snake_case Python field names

        """
        return {
            # Core fields that need manual processing
            "question_target": "",  # Always empty for EPPI
            "output_data_type": AttributeType.BOOL,  # Always boolean for EPPI
            "attribute_id": attr_data.get("AttributeId", 0),  # Convert int to str
            "attribute_label": attr_data.get("AttributeName", ""),
            # Explicitly map camelCase JSON fields to snake_case Python fields
            # (alias generators don't work in reverse for deserialization)
            "attribute_description": attr_data.get("AttributeDescription"),
            "attribute_type": attr_data.get("AttributeType"),
            "attribute_set_description": attr_data.get("AttributeSetDescription"),
            # hierarchy_path is already in snake_case from flatten_attributes_hierarchy
            # hierarchy_level is already in snake_case from flatten_attributes_hierarchy
            # is_leaf is already in snake_case from flatten_attributes_hierarchy
        }

    def process_document_data_for_validation(
        self, document_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Process raw document data for EppiDocument validation.

        Only handles fields that need manual processing -
        alias generators handle the rest.

        Args:
            document_data: Raw document data from EPPI JSON

        Returns:
            Dictionary with only the fields that need manual processing

        """
        return {
            # Core fields that need manual processing
            "name": document_data.get("Title"),  # Maps from "Title"
            "citation": self._create_reference(
                document_data
            ),  # Complex object creation
            "context": document_data.get("Abstract"),  # Maps from "Abstract"
            "document_id": str(document_data.get("ItemId", "")),  # Convert int to str
            "filename": document_data.get("Title", "").replace(" ", "_") + ".pdf"
            if document_data.get("Title")
            else None,
            # Note: All EPPI-specific fields (item_id, title, parent_title, etc.)
            # are automatically mapped by alias generators from camelCase JSON
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
            # Create a flattened version of this attribute
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

            # Add to flattened list
            flattened.append(flattened_attr)

            # Recursively process children if they exist
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
            # Get fields that need manual processing
            # (includes explicit camelCase to snake_case mapping)
            manual_fields = self.process_attribute_data_for_validation(attr_data)

            # Merge manual fields with raw data
            # manual_fields contains snake_case versions of all needed fields
            # attr_data may contain both camelCase and snake_case keys
            # We prioritize manual_fields (snake_case) and keep hierarchy
            # fields from attr_data
            
            # attr_data: Includes hierarchy_path, hierarchy_level, is_leaf
            # (already snake_case)
            
            # manual_fields: Overrides with snake_case versions of camelCase fields
            combined_data = {
                **attr_data,
                **manual_fields,
            }

            # Create the model using model_validate for proper alias handling
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
        # Get fields that need manual processing
        manual_fields = self.process_document_data_for_validation(document_data)

        # Merge manual fields with raw data (alias generators handle the rest)
        combined_data = {**document_data, **manual_fields}

        # Create the model - alias generators automatically map camelCase fields
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

            # Create EppiItemAttributeFullTextDetails object
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
        attributes_lookup: dict[str, EppiAttribute] | None = None,
        attribute_id_to_label: dict[str, str] | None = None,
    ) -> EppiGoldStandardAnnotation:
        """
        Convert a single annotation dictionary to EppiGoldStandardAnnotation.

        Args:
            annotation: Single annotation dictionary from EPPI JSON
            attributes_lookup: Lookup dictionary for attributes
            attribute_id_to_label: Mapping from attribute ID to label

        Returns:
            EppiGoldStandardAnnotation model

        """
        # Process text details
        text_details = annotation.get("ItemAttributeFullTextDetails", [])
        extracted_texts, item_attribute_details = self._process_text_details(
            text_details
        )

        # Join all extracted texts
        output_data = " | ".join(extracted_texts) if extracted_texts else ""

        # Look up the attribute from the attributes list
        attribute_id = str(annotation.get("AttributeId", ""))
        attribute = attributes_lookup.get(attribute_id) if attributes_lookup else None

        if not attribute:
            # Create a basic attribute if not found in lookup
            # Use the mapping to get the correct label
            attribute_label = (
                attribute_id_to_label.get(attribute_id, f"Attribute {attribute_id}")
                if attribute_id_to_label
                else f"Attribute {attribute_id}"
            )
            # Create minimal attribute data and process it
            minimal_attr_data = {
                "AttributeId": attribute_id,
                "AttributeName": attribute_label,
            }
            # Get fields that need manual processing
            manual_fields = self.process_attribute_data_for_validation(
                minimal_attr_data
            )

            # Merge manual fields with raw data (alias generators handle the rest)
            combined_data = {**minimal_attr_data, **manual_fields}

            # Create the model - alias generators automatically map camelCase fields
            attribute = EppiAttribute.model_validate(combined_data)
        # Ensure the attribute has the correct label from the mapping
        elif attribute_id_to_label and attribute_id in attribute_id_to_label:
            attribute.attribute_label = attribute_id_to_label[attribute_id]

        return EppiGoldStandardAnnotation(
            attribute=attribute,
            additional_text=annotation.get("AdditionalText", ""),
            arm_id=annotation.get("ArmId"),
            arm_title=annotation.get("ArmTitle", ""),
            arm_description=annotation.get("ArmDescription", ""),
            output_data=bool(
                output_data
            ),  # Convert to boolean which is the output data type for EPPI
            annotation_type=AnnotationType.HUMAN,  # All annotations from JSON are human
            item_attribute_full_text_details=item_attribute_details,
        )

    def convert_to_eppi_annotations(
        self,
        annotations_data: list[dict[str, Any]],
        document: EppiDocument,
        attributes_lookup: dict[str, EppiAttribute] | None = None,
        attribute_id_to_label: dict[str, str] | None = None,
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

    def _create_pdf_to_title_mapping(
        self, references: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Create mapping from PDF filenames to document titles."""
        pdf_to_title_mapping = {}
        for ref in references:
            title = ref.get("Title", "")
            # Create a potential PDF filename from the title
            pdf_filename = title.replace(" ", "_") + ".pdf"
            pdf_to_title_mapping[pdf_filename] = title

            # Also try with year if available
            year = ref.get("Year", "")
            if year:
                authors = (
                    ref.get("Authors", "").split(";")[0].strip()
                    if ref.get("Authors")
                    else ""
                )
                if authors:
                    # Try "Author Year.pdf" pattern
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
                # Direct title match
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

        data = self.load_eppi_json_annotations(file_path)
        raw_data = EppiRawData.model_validate(data)

        all_attributes_raw = self._extract_attributes_from_codesets(raw_data)

        attributes = self.convert_to_eppi_attributes(all_attributes_raw)

        attributes_lookup = {attr.attribute_id: attr for attr in attributes}

        attribute_id_to_label = {
            int(attr.attribute_id): attr.attribute_label for attr in attributes
        }

        # extract annotations from References
        all_annotations_raw = []
        documents_by_title = {}

        for reference in data.get("References", []):
            # The actual annotations are in References[].Codes
            reference_codes = reference.get("Codes", [])
            all_annotations_raw.extend(reference_codes)

            doc_title = reference.get("Title", "")
            if doc_title and doc_title not in documents_by_title:
                document = self.convert_to_eppi_document(reference)
                documents_by_title[doc_title] = document

        # Create a mapping from PDF filenames to document titles
        pdf_to_title_mapping = self._create_pdf_to_title_mapping(
            data.get("References", [])
        )

        # Convert all annotations, linking them to their respective documents
        annotated_documents = []
        all_annotations = []

        for doc_title, document in documents_by_title.items():
            # Get annotations for this specific document
            doc_annotations = self._find_document_annotations(
                all_annotations_raw, doc_title, pdf_to_title_mapping
            )

            if (
                doc_annotations
            ):  # Only process if there are annotations for this document
                annotations = self.convert_to_eppi_annotations(
                    doc_annotations,
                    document,
                    attributes_lookup,  # type: ignore[arg-type]
                    attribute_id_to_label,  # type: ignore[arg-type]
                )

                # Create EppiGoldStandardAnnotatedDocument
                # Since it inherits from EppiDocument, we pass all document fields
                annotated_doc = EppiGoldStandardAnnotatedDocument(
                    **document.model_dump(), annotations=annotations
                )
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
        # Create the output directory structure
        base_path = Path(output_dir)

        # Always create an 'eppi' subdirectory
        eppi_base_path = base_path  # / "eppi"

        # If input_filename, create sub-dir with the filename (without extension)
        if input_filename:
            filename_without_ext = Path(input_filename).stem
            eppi_path = eppi_base_path / filename_without_ext
        else:
            eppi_path = eppi_base_path

        eppi_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save each collection as JSON model_dump_json()
        # file_mappings = [
        #     ("attributes", processed_data.attributes),
        #     ("documents", processed_data.documents),
        #     ("annotated_documents", processed_data.annotated_documents),
        # ]

        file_mappings = {
            "attributes": processed_data.attributes,
            "documents": processed_data.documents,
            "annotated_documents": processed_data.annotated_documents,
        }

        for file_type, data_list in file_mappings.items():
            # for item in data_list:  # type: ignore[attr-defined]
            #     logger.debug(item)
            #     logger.debug(type(item))
            #     logger.debug(item.model_dump_json())
            file_path = eppi_path / f"{file_type}.json"
            file_path.write_text(
                json.dumps(
                    [item.model_dump(mode="json") for item in data_list],  # type: ignore[attr-defined]
                    indent=2,
                )
            )
            saved_files[file_type] = str(file_path)

        # Save attribute mapping as simple JSON
        mapping_file = eppi_path / "attribute_id_to_label_mapping.json"
        mapping_file.write_text(
            json.dumps(processed_data.attribute_id_to_label, indent=2)
        )
        saved_files["attribute_mapping"] = str(mapping_file)

        return saved_files
