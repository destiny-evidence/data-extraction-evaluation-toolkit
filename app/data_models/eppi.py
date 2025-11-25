"""EPPI-specific data models extending the core models."""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    Document,
    GoldStandardAnnotation,
)


class EppiAttribute(Attribute):
    """
    EPPI-specific attribute with additional fields.

    Extends the core Attribute class with EPPI-specific
    metadata and hierarchy information.

    Uses alias generators to automatically map
    camelCase EPPI JSON fields to snake_case Python fields.
    """

    model_config = ConfigDict(alias_generators=to_camel)  # type: ignore[typeddict-unknown-key]

    # Core fields (inherited from Attribute) - these need manual processing
    question_target: str = ""  # Always empty for EPPI
    output_data_type: AttributeType = AttributeType.BOOL

    # EPPI-specific fields - these map automatically from camelCase JSON
    attribute_set_description: str | None = Field(
        description="Description of the attribute set this attribute belongs to",
        default=None,
    )
    hierarchy_path: str | None = Field(
        description="Dot-separated path showing the hierarchical "
        " position of this attribute",
        default=None,
    )
    hierarchy_level: int = Field(
        description="Numeric level indicating depth in "
        " the attribute hierarchy (0 = root level)",
        default=0,
    )
    is_leaf: bool = Field(
        description="Whether this attribute is a leaf node  (has no child attributes)",
        default=True,
    )
    parent_attribute_id: str | None = Field(
        description="ID of the parent attribute in the hierarchy", default=None
    )
    attribute_type: str | None = Field(
        description="Whether the attribute is Selectable in the "
        " EPPI-Reviewer interface or not",
        default=None,
    )
    attribute_description: str | None = Field(
        description="Detailed description explaining what this attribute represents",
        default=None,
    )


class EppiDocument(Document):
    """
    EPPI-specific document.

    Uses alias generators to automatically map
    camelCase EPPI JSON fields to snake_case Python fields.
    """

    model_config = ConfigDict(alias_generators=to_camel)  # type: ignore[typeddict-unknown-key]

    # EPPI-specific fields - these map automatically from camelCase JSON
    item_id: int | None = None
    title: str | None = None
    parent_title: str | None = None
    short_title: str | None = None
    date_created: str | None = None
    edited_by: str | None = None
    year: str | None = None
    month: str | None = None
    abstract: str | None = None
    authors: str | None = None
    keywords: str | None = None
    doi: str | None = None


class EppiItemAttributeFullTextDetails(BaseModel):
    """
    EPPI-specific item attribute full text details.

    Arm specific information, exact text keywords for the attribute.
    """

    item_document_id: int | None = None
    text: str | None = None
    item_arm: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_at_least_one_field(cls, data: dict) -> dict:
        """Ensure at least one field is not None."""
        if all(v is None for k, v in data.items()):
            msg = (
                "At least one field must be provided "
                "(item_document_id, text, or item_arm)"
            )
            raise ValueError(msg)
        return data


class EppiGoldStandardAnnotation(GoldStandardAnnotation):
    """
    EPPI-specific gold standard annotation.

    In EPPI-Reviewer context, an "arm" refers to a study group or intervention group
    within a research study (e.g., "Treatment Group", "Control Group", "Placebo Group").
    Each annotation is associated with a specific arm to indicate which study group
    the extracted information relates to.
    """

    attribute: EppiAttribute = Field(
        description="The EPPI attribute being annotated  "
        "with hierarchy and metadata info."
    )
    additional_text: str | None = Field(
        description="Notes provided by the annotator - usually the citation "
        " from the paper containing the context window where the attribute is found",
        default=None,
    )
    arm_id: int | None = Field(
        description="ID of the study arm this annotation relates to", default=None
    )
    arm_title: str | None = Field(
        description="Title or name of the study arm", default=None
    )
    arm_description: str | None = Field(
        description="Detailed description of the study arm", default=None
    )
    item_attribute_full_text_details: list[EppiItemAttributeFullTextDetails] | None = (
        Field(
            description="List of detailed text extracts and "
            " arm-specific information for this annotation",
            default=None,
        )
    )

    # additional, optional llm-based fields
    reasoning: str | None = Field(
        description="Reasoning, taken from LLM response", default=None
    )


class EppiGoldStandardAnnotatedDocument(EppiDocument):
    """EPPI-specific gold standard annotated document."""

    annotations: list[EppiGoldStandardAnnotation]


class EppiCodeSet(BaseModel):
    """
    Represents a single CodeSet from EPPI JSON.

    CodeSets contain hierarchical attribute definitions used in EPPI-Reviewer.
    """

    attributes: dict[str, Any] | None = Field(alias="Attributes", default=None)

    def get_attributes_list(self) -> list[dict[str, Any]]:
        """Extract AttributesList from the CodeSet."""
        if self.attributes and "AttributesList" in self.attributes:
            return self.attributes["AttributesList"]
        return []


class EppiRawData(BaseModel):
    """
    Represents the complete EPPI JSON structure.

    This model validates and structures the raw EPPI JSON data,
    making it easier to work with and validate.
    """

    code_sets: list[EppiCodeSet] = Field(alias="CodeSets", default=[])
    references: list[dict[str, Any]] = Field(alias="References", default=[])

    def extract_all_attributes(
        self, flatten_hierarchy_func: Callable[[list], list]
    ) -> list[dict[str, Any]]:
        """
        Extract and flatten attributes from all CodeSets.

        Args:
            flatten_hierarchy_func: Function to flatten attribute hierarchy

        Returns:
            List of flattened attribute dictionaries

        """
        all_attributes = []
        for codeset in self.code_sets:
            attributes_list = codeset.get_attributes_list()
            if attributes_list:
                flattened = flatten_hierarchy_func(attributes_list)
                all_attributes.extend(flattened)
        return all_attributes


class ProcessedAnnotationData(BaseModel):
    """
    Structured result from annotation processing.

    This model provides a clean, validated structure for all processed
    annotation data with useful properties and methods.
    """

    attributes: list[EppiAttribute]
    documents: list[EppiDocument]
    annotations: list[EppiGoldStandardAnnotation]
    annotated_documents: list[EppiGoldStandardAnnotatedDocument]
    attribute_id_to_label: dict[int, str]
    raw_data: EppiRawData

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
            bad_filetype = "file ending must be .csv."
            raise ValueError(bad_filetype)
        for attribute in self.attributes:
            attribute.write_to_csv(filepath=filepath)

        logger.debug(f"wrote attributes to file {filepath}.")

    def _import_prompts_csv_file(
        self, filepath: Path, *, overwrite: bool = True
    ) -> None:
        """
        Import prompts from a csv file.

        Args:
            filepath (Path): attribute/prompt input file.
            overwrite (bool, optional): Overwrite existing prompts. Defaults to True.

        """
        if not filepath.exists():
            no_file = f"CSV file not found: {filepath}"
            raise FileNotFoundError(no_file)

        if filepath.suffix != ".csv":
            bad_suffix = "File must have .csv extension"
            raise ValueError(bad_suffix)

        with filepath.open(mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                empty_csv = "CSV file is empty or has no headers"
                raise ValueError(empty_csv)

            required_fields = ["attribute_id", "prompt"]
            for field in required_fields:
                if field not in reader.fieldnames:
                    csv_missing_fields = (
                        f"CSV must contain '{field}' column. "
                        f"Found columns: {', '.join(reader.fieldnames)}"
                    )

                    raise ValueError(csv_missing_fields)

            rows_processed = 0
            for row in reader:
                # find attribute_id match
                attribute_id = int(row["attribute_id"])
                matching_attribute = None

                for attribute in self.attributes:
                    if attribute.attribute_id == attribute_id:
                        matching_attribute = attribute
                        break

                if matching_attribute is None:
                    logger.warning(
                        f"No attribute found with ID {attribute_id}, skipping row"
                    )
                    continue

                # populate prompt using the Attribute method
                try:
                    matching_attribute.populate_prompt_from_dict(
                        row, overwrite=overwrite
                    )
                    rows_processed += 1
                except ValueError as e:
                    logger.error(
                        f"Error processing row for attribute {attribute_id}: {e}"
                    )

            logger.info(f"Processed {rows_processed} prompts from {filepath}")

    def populate_custom_prompts(
        self, method: Literal["cli", "file"], filepath: Path | None = None, **kwargs
    ) -> None:
        """
        Populate custom prompts.

        Args:
            method (Literal[&quot;cli&quot;, &quot;file&quot;])
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

    def get_attributes_by_type(self, attribute_type: str) -> list[EppiAttribute]:
        """Get all attributes of a specific type."""
        return [
            attr for attr in self.attributes if attr.attribute_type == attribute_type
        ]

    def get_documents_with_annotations(self) -> list[EppiDocument]:
        """Get only documents that have annotations."""
        annotated_doc_ids = {doc.document_id for doc in self.annotated_documents}
        return [doc for doc in self.documents if doc.document_id in annotated_doc_ids]

    def get_annotations_by_type(
        self, annotation_type: AnnotationType
    ) -> list[EppiGoldStandardAnnotation]:
        """Get all annotations of a specific type (human/llm)."""
        return [
            ann for ann in self.annotations if ann.annotation_type == annotation_type
        ]

    def get_attribute_by_id(self, attribute_id: str) -> EppiAttribute | None:
        """Get an attribute by its ID."""
        for attr in self.attributes:
            if attr.attribute_id == attribute_id:
                return attr
        return None


class AttributeAnswerCoT(BaseModel):
    """Detailed answer format for a single attribute with reasoning."""

    attribute_name: str = Field(
        description="The name of the attribute being asked about"
    )
    answer: str = Field(description="The answer to the question, 'True' or 'False'")
    reasoning: str = Field(description="The reasoning behind the answer")
    citation: str | None = Field(
        description="The citation from the Research Information to support the answer"
    )


class BatchAnswerFormatCoT(BaseModel):
    """Batch answers for all attributes with reasoning."""

    answers: list[AttributeAnswerCoT] = Field(
        description="List of answers for each attribute"
    )
