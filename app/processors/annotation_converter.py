"""Convert annotation JSON files to Pydantic models."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from destiny_sdk.references import Reference

from app.logger import logger
from app.models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
)

# Rebuild models to resolve forward references
EppiDocument.model_rebuild()


class AnnotationConverter:
    """
    A class to convert raw EPPI-Reviewer JSON annotations into structured Pydantic models.

    This converter handles the complex hierarchical structure of EPPI attributes by flattening
    them while preserving parent-child relationships through path information.
    """

    def load_json_annotations(self, file_path: str | Path) -> dict[str, Any]:
        """
        Load JSON annotations from a file.

        Args:
            file_path: Path to the JSON annotation file

        Returns:
            Dictionary containing the loaded annotations

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
                "AttributeDescription": attr.get("AttributeDescription", ""),
                "AttributeSetDescription": attr.get("AttributeSetDescription", ""),
                "AttributeType": attr.get("AttributeType", ""),
                "AttributeTypeId": attr.get("AttributeTypeId"),
                "AttributeSetId": attr.get("AttributeSetId"),
                "OriginalAttributeID": attr.get("OriginalAttributeID"),
                "ExtURL": attr.get("ExtURL", ""),
                "ExtType": attr.get("ExtType", ""),
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
            # Create the attribute with proper mapping
            attribute = EppiAttribute(
                question_target="",  # Leave empty as requested
                output_data_type="bool",  # All EPPI attributes are boolean as requested
                attribute_id=str(attr_data.get("AttributeId", "")),
                attribute_label=attr_data.get("AttributeName", ""),
                attribute_set_description=attr_data.get("AttributeSetDescription", ""),
                # Additional hierarchy information
                hierarchy_path=attr_data.get("hierarchy_path", ""),
                hierarchy_level=attr_data.get("hierarchy_level", 0),
                is_leaf=attr_data.get("is_leaf", True),
                parent_attribute_id=str(attr_data.get("AttributeSetId", "")),
                attribute_type=attr_data.get("AttributeType", ""),
                attribute_description=attr_data.get("AttributeDescription", ""),
            )
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
        return EppiDocument(
            name=document_data.get("Title", ""),
            citation=Reference(
                id=str(uuid.uuid4()),
                title=document_data.get("Title", ""),
                authors=[
                    {"full_name": author.strip()}
                    for author in document_data.get("Authors", "").split(";")
                    if author.strip()
                ],
                year=int(document_data.get("Year", 0))
                if document_data.get("Year") is not None
                else None,
            ),
            context=document_data.get("Abstract", ""),
            document_id=str(document_data.get("ItemId", "")),
            filename=document_data.get("Title", "").replace(" ", "_") + ".pdf",
            # EPPI-specific fields
            item_id=int(document_data.get("ItemId", 0))
            if document_data.get("ItemId") is not None
            else None,
            title=document_data.get("Title", ""),
            parent_title=document_data.get("ParentTitle", ""),
            short_title=document_data.get("ShortTitle", ""),
            date_created=document_data.get("DateCreated", ""),
            edited_by=document_data.get("EditedBy", ""),
            year=document_data.get("Year", ""),
            month=document_data.get("Month", ""),
            abstract=document_data.get("Abstract", ""),
            authors=document_data.get("Authors", ""),
            keywords=document_data.get("Keywords", ""),
            doi=document_data.get("DOI", ""),
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

        Returns:
            List of EppiGoldStandardAnnotation models

        """
        annotations = []

        for annotation in annotations_data:
            # Extract text from ItemAttributeFullTextDetails
            extracted_texts = []
            item_attribute_details = []

            for text_detail in annotation.get("ItemAttributeFullTextDetails", []):
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

            # Join all extracted texts
            output_data = " | ".join(extracted_texts) if extracted_texts else ""

            # Look up the attribute from the attributes list
            attribute_id = str(annotation.get("AttributeId", ""))
            attribute = (
                attributes_lookup.get(attribute_id) if attributes_lookup else None
            )

            if not attribute:
                # Create a basic attribute if not found in lookup
                # Use the mapping to get the correct label
                attribute_label = (
                    attribute_id_to_label.get(attribute_id, f"Attribute {attribute_id}")
                    if attribute_id_to_label
                    else f"Attribute {attribute_id}"
                )
                attribute = EppiAttribute(
                    question_target="",
                    output_data_type="bool",
                    attribute_id=attribute_id,
                    attribute_label=attribute_label,
                )
            # Ensure the attribute has the correct label from the mapping
            elif attribute_id_to_label and attribute_id in attribute_id_to_label:
                attribute.attribute_label = attribute_id_to_label[attribute_id]

            annotation_model = EppiGoldStandardAnnotation(
                attribute=attribute,
                additional_text=annotation.get("AdditionalText", ""),
                arm_id=annotation.get("ArmId"),
                arm_title=annotation.get("ArmTitle", ""),
                arm_description=annotation.get("ArmDescription", ""),
                output_data=bool(output_data),  # Convert to boolean as requested
                item_attribute_full_text_details=item_attribute_details,
            )
            annotations.append(annotation_model)

        return annotations

    def _extract_attributes_from_codesets(
        self, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract and flatten attributes from CodeSets."""
        all_attributes_raw = []

        # Process CodeSets[0] (first CodeSet) - usually contains basic attributes like "Arm name"
        if len(data.get("CodeSets", [])) > 0:
            codeset0 = data["CodeSets"][0]
            if "Attributes" in codeset0 and "AttributesList" in codeset0["Attributes"]:
                attributes_list0 = codeset0["Attributes"]["AttributesList"]
                all_attributes_raw.extend(
                    self.flatten_attributes_hierarchy(attributes_list0)
                )

        # Process CodeSets[1] (second CodeSet) - contains the main hierarchical attributes
        if len(data.get("CodeSets", [])) > 1:
            codeset1 = data["CodeSets"][1]
            if "Attributes" in codeset1 and "AttributesList" in codeset1["Attributes"]:
                attributes_list1 = codeset1["Attributes"]["AttributesList"]
                all_attributes_raw.extend(
                    self.flatten_attributes_hierarchy(attributes_list1)
                )

        return all_attributes_raw

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

    def process_annotation_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Process a complete annotation file and return structured data.

        Args:
            file_path: Path to the JSON annotation file

        Returns:
            Dictionary containing processed attributes, documents, and annotations

        """
        # Load the JSON data
        data = self.load_json_annotations(file_path)

        # Extract and flatten attributes from both CodeSets
        all_attributes_raw = self._extract_attributes_from_codesets(data)

        # Convert to Pydantic models
        attributes = self.convert_to_eppi_attributes(all_attributes_raw)

        # Create attributes lookup for annotation processing
        attributes_lookup = {attr.attribute_id: attr for attr in attributes}

        # Create attribute ID to label mapping
        attribute_id_to_label = {
            attr.attribute_id: attr.attribute_label for attr in attributes
        }

        # Extract annotations from References
        all_annotations_raw = []
        documents_by_title = {}

        for reference in data.get("References", []):
            # The actual annotations are in References[].Codes
            reference_codes = reference.get("Codes", [])
            all_annotations_raw.extend(reference_codes)

            # Extract document info from the reference itself
            doc_title = reference.get("Title", "")
            if doc_title and doc_title not in documents_by_title:
                # Create a basic document entry
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
                    doc_annotations, document, attributes_lookup, attribute_id_to_label
                )

                # Create EppiGoldStandardAnnotatedDocument
                annotated_doc = EppiGoldStandardAnnotatedDocument(
                    document=document, annotations=annotations
                )
                annotated_documents.append(annotated_doc)
                all_annotations.extend(annotations)

        return {
            "attributes": attributes,
            "documents": list(documents_by_title.values()),
            "annotations": all_annotations,
            "annotated_documents": annotated_documents,
            "attribute_id_to_label": attribute_id_to_label,
            "raw_data": data,
        }

    def save_processed_data(
        self,
        processed_data: dict[str, Any],
        output_dir: str | Path = "app/annotations/processed",
    ) -> dict[str, str]:
        """
        Save processed data to structured files.

        Args:
            processed_data: The processed data from process_annotation_file
            output_dir: Directory to save the processed files

        Returns:
            Dictionary mapping data types to saved file paths

        """
        # Create the EPPI subdirectory
        eppi_path = Path(output_dir)
        eppi_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Custom JSON encoder to handle UUIDs and other non-serializable objects
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj: object) -> str:
                if hasattr(obj, "__str__"):
                    return str(obj)
                return super().default(obj)

        # Save attributes
        attributes_file = eppi_path / "attributes.json"
        with attributes_file.open("w", encoding="utf-8") as f:
            json.dump(
                [attr.model_dump() for attr in processed_data["attributes"]],
                f,
                indent=2,
                cls=CustomJSONEncoder,
            )
        saved_files["attributes"] = str(attributes_file)

        # Save documents
        documents_file = eppi_path / "documents.json"
        with documents_file.open("w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump() for doc in processed_data["documents"]],
                f,
                indent=2,
                cls=CustomJSONEncoder,
            )
        saved_files["documents"] = str(documents_file)

        # Save annotated documents (documents with their annotations) - this is the main file
        annotated_docs_file = eppi_path / "annotated_documents.json"
        with annotated_docs_file.open("w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump() for doc in processed_data["annotated_documents"]],
                f,
                indent=2,
                cls=CustomJSONEncoder,
            )
        saved_files["annotated_documents"] = str(annotated_docs_file)

        # Save attribute ID to label mapping
        attribute_mapping_file = eppi_path / "attribute_id_to_label_mapping.json"
        with attribute_mapping_file.open("w", encoding="utf-8") as f:
            json.dump(processed_data["attribute_id_to_label"], f, indent=2)
        saved_files["attribute_mapping"] = str(attribute_mapping_file)

        return saved_files


def main() -> None:
    """Run the annotation converter."""
    parser = argparse.ArgumentParser(
        description="Convert EPPI annotations to structured format"
    )
    parser.add_argument("input_file", help="Path to the raw EPPI JSON file")
    parser.add_argument("output_dir", help="Directory to save processed files")
    args = parser.parse_args()
    converter = AnnotationConverter()
    processed_data = converter.process_annotation_file(args.input_file)
    saved_files = converter.save_processed_data(processed_data, args.output_dir)
    logger.info("Conversion complete!")
    for file_type, file_path in saved_files.items():
        logger.info(f"  {file_type}: {file_path}")


if __name__ == "__main__":
    main()
