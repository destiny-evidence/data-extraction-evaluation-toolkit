"""Generalizable data extraction module for LLM-based document analysis."""

import json
import os
from enum import Enum
from pathlib import Path

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.logger import logger
from app.models.base import AnnotationType
from app.models.eppi import EppiAttribute, EppiDocument, EppiGoldStandardAnnotation


class ContextType(str, Enum):
    """Types of context that can be provided to the LLM."""

    FULL_DOCUMENT = "full_document"
    ABSTRACT_ONLY = "abstract_only"
    RAG_SNIPPETS = "rag_snippets"
    CUSTOM = "custom"


class AttributeSelectionMode(str, Enum):
    """Modes for selecting attributes to extract."""

    ALL = "all"
    SINGLE = "single"
    BATCH = "batch"
    BY_NAMES = "by_names"
    BY_IDS = "by_ids"


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    system_prompt: str = Field(
        description="System prompt that defines the task and role",
        default="You are an expert research analyst. Provide detailed, accurate analysis.",
    )
    attribute_specific_prompt: str = Field(
        description="Prompt template for attribute-specific extraction",
        default="Analyze this research document and answer questions about specific attributes. For each attribute, determine if it's present (True/False), provide reasoning, and include citations.",
    )
    single_attribute_prompt: str = Field(
        description="Prompt template for single attribute extraction",
        default="Analyze this research document for the following attribute: {attribute_label}. Determine if it's present (True/False), provide reasoning, and include citations.",
    )
    batch_attribute_prompt: str = Field(
        description="Prompt template for batch attribute extraction",
        default="Analyze this research document and answer questions about the following attributes. For each attribute, determine if it's present (True/False), provide reasoning, and include citations.",
    )


class DataExtractionConfig(BaseModel):
    """Configuration for data extraction tasks."""

    # LLM Configuration
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    temperature: float = Field(
        default=0.1, description="Temperature for LLM generation"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum tokens for LLM response"
    )

    # Context Configuration
    context_type: ContextType = Field(
        default=ContextType.FULL_DOCUMENT, description="Type of context to provide"
    )
    max_context_length: int = Field(
        default=40000, description="Maximum context length for LLM"
    )

    # Attribute Selection
    attribute_selection_mode: AttributeSelectionMode = Field(
        default=AttributeSelectionMode.ALL, description="Mode for selecting attributes"
    )
    selected_attribute_ids: list[str] = Field(
        default=[], description="Specific attribute IDs to extract"
    )
    selected_attribute_names: list[str] = Field(
        default=[], description="Specific attribute names to extract"
    )

    # Prompt Configuration
    prompt_config: PromptConfig = Field(
        default_factory=PromptConfig, description="Prompt configuration"
    )

    # Output Configuration
    include_reasoning: bool = Field(
        default=True, description="Include reasoning in output"
    )
    include_additional_text: bool = Field(
        default=True, description="Include additional text/citations in output"
    )


class DataExtractionModule:
    """
    Generalizable module for LLM-based data extraction from documents.

    This module provides a flexible interface for extracting structured data from documents
    using LLMs, with support for different context types, attribute selection modes, and
    customizable prompts.
    """

    def __init__(
        self,
        config: DataExtractionConfig | None = None,
        custom_system_prompt_file: Path | None = None,
    ) -> None:
        """
        Initialize the data extraction module.

        Args:
            config: Configuration for the extraction module. If None, uses default config.
            custom_system_prompt_file: Optional custom system prompt file path

        """
        self.config = config or DataExtractionConfig()
        self.custom_system_prompt_file = custom_system_prompt_file
        load_dotenv(override=True)

        # Set up LLM model
        if os.getenv("AZURE_API_KEY"):
            deployment = os.getenv("AZURE_DEPLOYMENT", self.config.model)
            self.model = f"azure/{deployment}"
        else:
            self.model = self.config.model

    def extract_from_documents(
        self,
        documents: list[EppiDocument],
        attributes: list[EppiAttribute],
        output_file: Path | None = None,
    ) -> list[EppiGoldStandardAnnotation]:
        """
        Extract data from multiple documents.

        Args:
            documents: List of documents to analyze
            attributes: List of attributes to extract
            output_file: Optional file to save results

        Returns:
            List of annotations for all documents and attributes

        """
        all_annotations = []

        for document in documents:
            logger.info(f"Processing document: {document.name}")
            document_annotations = self.extract_from_document(document, attributes)
            all_annotations.extend(document_annotations)

        if output_file:
            self._save_results(all_annotations, output_file)

        return all_annotations

    def extract_from_document(
        self,
        document: EppiDocument,
        attributes: list[EppiAttribute],
    ) -> list[EppiGoldStandardAnnotation]:
        """
        Extract data from a single document.

        Args:
            document: Document to analyze
            attributes: List of attributes to extract

        Returns:
            List of annotations for the document

        """
        # Filter attributes based on selection mode
        selected_attributes = self._filter_attributes(attributes)

        if not selected_attributes:
            logger.warning("No attributes selected for extraction")
            return []

        # Prepare context
        context = self._prepare_context(document)

        # Generate prompt
        prompt = self._generate_prompt(context, selected_attributes)

        # Call LLM
        llm_response = self._call_llm(prompt)

        # Parse response and create annotations
        return self._parse_llm_response(llm_response, selected_attributes, document)

    def _filter_attributes(
        self, attributes: list[EppiAttribute]
    ) -> list[EppiAttribute]:
        """Filter attributes based on selection mode."""
        if self.config.attribute_selection_mode == AttributeSelectionMode.ALL:
            return attributes

        if self.config.attribute_selection_mode == AttributeSelectionMode.SINGLE:
            if self.config.selected_attribute_ids:
                return [
                    attr
                    for attr in attributes
                    if attr.attribute_id == self.config.selected_attribute_ids[0]
                ]
            if self.config.selected_attribute_names:
                return [
                    attr
                    for attr in attributes
                    if attr.attribute_label == self.config.selected_attribute_names[0]
                ]
            return attributes[:1]  # Default to first attribute

        if self.config.attribute_selection_mode == AttributeSelectionMode.BATCH:
            if self.config.selected_attribute_ids:
                return [
                    attr
                    for attr in attributes
                    if attr.attribute_id in self.config.selected_attribute_ids
                ]
            if self.config.selected_attribute_names:
                return [
                    attr
                    for attr in attributes
                    if attr.attribute_label in self.config.selected_attribute_names
                ]
            return attributes[:5]  # Default to first 5 attributes

        if self.config.attribute_selection_mode == AttributeSelectionMode.BY_IDS:
            return [
                attr
                for attr in attributes
                if attr.attribute_id in self.config.selected_attribute_ids
            ]

        if self.config.attribute_selection_mode == AttributeSelectionMode.BY_NAMES:
            return [
                attr
                for attr in attributes
                if attr.attribute_label in self.config.selected_attribute_names
            ]

        return attributes

    def _prepare_context(self, document: EppiDocument) -> str:
        """Prepare context based on context type."""
        if self.config.context_type == ContextType.FULL_DOCUMENT:
            context = document.context
        elif self.config.context_type == ContextType.ABSTRACT_ONLY:
            context = document.abstract or document.context
        elif self.config.context_type == ContextType.RAG_SNIPPETS:
            # For now, use abstract as RAG snippets placeholder
            # In the future, this could be replaced with actual RAG snippets
            context = document.abstract or document.context
        else:  # CUSTOM
            context = document.context

        # Ensure context is a string
        if isinstance(context, list):
            context = " ".join(context)

        # Truncate if too long
        if len(context) > self.config.max_context_length:
            context = context[: self.config.max_context_length] + "..."

        return context

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        # Use custom prompt file if provided
        if self.custom_system_prompt_file:
            prompt_file = self.custom_system_prompt_file
        else:
            prompt_file = (
                Path(__file__).parent.parent / "prompts" / "system_prompt_v0.txt"
            )

        try:
            return prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(
                f"System prompt file not found at {prompt_file}, using default"
            )
            return self.config.prompt_config.system_prompt

    def _generate_prompt(self, context: str, attributes: list[EppiAttribute]) -> str:
        """Generate prompt based on attribute selection mode."""
        # Load system prompt from file
        system_prompt_template = self._load_system_prompt()

        if self.config.attribute_selection_mode == AttributeSelectionMode.SINGLE:
            attribute = attributes[0]
            # Use attribute_set_description as the question
            question = attribute.attribute_set_description or "No description available"

            # Format the system prompt with context and question
            formatted_prompt = system_prompt_template.format(
                context=context, question=question
            )
        else:
            # For batch mode, create a list of questions from attribute descriptions
            questions = []
            for attr in attributes:
                question_text = f"- {attr.attribute_label} (ID: {attr.attribute_id}): {attr.attribute_set_description or 'No description'}"
                questions.append(question_text)

            question = "\n".join(questions)

            # Format the system prompt with context and questions
            formatted_prompt = system_prompt_template.format(
                context=context, question=question
            )

        # Add JSON response format instructions
        return f"""
{formatted_prompt}

Respond in JSON format:
{{
    "annotations": [
        {{
            "attribute_id": "attribute_id",
            "output_data": true or false,
            "annotation_type": "llm",
            "additional_text": "Supporting text from document (or null)",
            "reasoning": "Your reasoning for the decision"
        }}
    ]
}}
"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # Load system prompt from file
        system_prompt = self._load_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        logger.info("=" * 60)
        logger.info("LLM REQUEST")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model}")
        logger.info(f"Temperature: {self.config.temperature}")
        logger.info("Response Format: JSON Object")
        logger.info("")
        logger.info("System Message:")
        logger.info(messages[0]["content"])
        logger.info("")
        logger.info("User Message:")
        logger.info(messages[1]["content"])
        logger.info("=" * 60)

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            max_tokens=self.config.max_tokens,
        )

        response_content = response.choices[0].message.content
        logger.info("")
        logger.info("LLM RESPONSE")
        logger.info("=" * 60)
        logger.info("Raw Response:")
        logger.info(response_content)
        logger.info("=" * 60)

        return response_content

    def _parse_llm_response(
        self,
        response_content: str,
        attributes: list[EppiAttribute],
        document: EppiDocument,
    ) -> list[EppiGoldStandardAnnotation]:
        """Parse LLM response and create annotations."""
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return []

        annotations = []
        for annotation_data in result.get("annotations", []):
            attribute_id = annotation_data.get("attribute_id")
            attribute = next(
                (attr for attr in attributes if attr.attribute_id == attribute_id), None
            )

            if attribute:
                annotation = EppiGoldStandardAnnotation(
                    attribute=attribute,
                    output_data=annotation_data.get("output_data", False),
                    annotation_type=AnnotationType.LLM,
                    additional_text=annotation_data.get("additional_text")
                    if self.config.include_additional_text
                    else None,
                    reasoning=annotation_data.get("reasoning")
                    if self.config.include_reasoning
                    else None,
                )
                annotations.append(annotation)

        return annotations

    def _save_results(
        self, annotations: list[EppiGoldStandardAnnotation], output_file: Path
    ) -> None:
        """Save results to file."""
        annotations_data = []
        for annotation in annotations:
            annotation_dict = annotation.model_dump()
            # Convert type objects to strings for JSON serialization
            if (
                "attribute" in annotation_dict
                and "output_data_type" in annotation_dict["attribute"]
            ):
                annotation_dict["attribute"]["output_data_type"] = str(
                    annotation_dict["attribute"]["output_data_type"]
                )
            annotations_data.append(annotation_dict)

        with output_file.open("w") as f:
            json.dump({"annotations": annotations_data}, f, indent=2)

        logger.info(f"Results saved to: {output_file}")


# Convenience functions for common use cases
def extract_single_attribute(
    document: EppiDocument,
    attribute: EppiAttribute,
    config: DataExtractionConfig | None = None,
) -> EppiGoldStandardAnnotation | None:
    """Extract a single attribute from a document."""
    module = DataExtractionModule(config)
    annotations = module.extract_from_document(document, [attribute])
    return annotations[0] if annotations else None


def extract_all_attributes(
    document: EppiDocument,
    attributes: list[EppiAttribute],
    config: DataExtractionConfig | None = None,
) -> list[EppiGoldStandardAnnotation]:
    """Extract all attributes from a document."""
    module = DataExtractionModule(config)
    return module.extract_from_document(document, attributes)


def extract_batch_attributes(
    document: EppiDocument,
    attributes: list[EppiAttribute],
    attribute_ids: list[str],
    config: DataExtractionConfig | None = None,
) -> list[EppiGoldStandardAnnotation]:
    """Extract a batch of specific attributes from a document."""
    if config is None:
        config = DataExtractionConfig()

    config.attribute_selection_mode = AttributeSelectionMode.BY_IDS
    config.selected_attribute_ids = attribute_ids

    module = DataExtractionModule(config)
    return module.extract_from_document(document, attributes)
