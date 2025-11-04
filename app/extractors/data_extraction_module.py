"""Generalisable data extraction module for LLM-based document analysis."""

import json
import os
from enum import Enum
from pathlib import Path

import litellm
from pydantic import BaseModel, Field, ValidationError

from app.logger import logger
from app.models.base import AnnotationType
from app.models.eppi import (
    EppiAttribute,
    EppiDocument,
    EppiGoldStandardAnnotation,
    LLMResponseSchema,
)  # type: ignore[attr-defined]
from app.settings import get_settings

# Load centralized settings once for module-level defaults
settings = get_settings()


class ContextType(str, Enum):
    """Types of context that can be provided to the LLM."""

    FULL_DOCUMENT = "full_document"
    ABSTRACT_ONLY = "abstract_only"
    RAG_SNIPPETS = "rag_snippets"
    CUSTOM = "custom"


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    system_prompt: str = Field(
        description="System prompt that defines the task and role",
        default=(
            "You are an expert research analyst. Provide detailed, "
            "accurate analysis."
        ),
    )
    attribute_specific_prompt: str = Field(
        description="Prompt template for attribute-specific extraction",
        default=(
            "Analyze this research document and answer questions about "
            "specific attributes. For each attribute, determine if it's "
            "present (True/False), provide reasoning, and include citations."
        ),
    )
    single_attribute_prompt: str = Field(
        description="Prompt template for single attribute extraction",
        default=(
            "Analyze this research document for the following attribute: "
            "{attribute_label}. Determine if it's present (True/False), "
            "provide reasoning, and include citations."
        ),
    )
    batch_attribute_prompt: str = Field(
        description="Prompt template for batch attribute extraction",
        default=(
            "Analyze this research document and answer questions about the "
            "following attributes. For each attribute, determine if it's "
            "present (True/False), provide reasoning, and include citations."
        ),
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

    selected_attribute_ids: list[str] = Field(
        default=[], description="Specific attribute IDs to extract"
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
    Generalisable module for LLM-based data extraction from documents.

    This module provides a flexible interface for extracting structured data
    from documents using LLMs, with support for different context types and
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
            config: Configuration for the extraction module.
                If None, uses default config from centralized settings.
            custom_system_prompt_file: Optional custom system prompt file path.
                If None, defaults to `app/prompts/system_prompt_v0.txt`.
                If that file doesn't exist, falls back to the prompt
                from `config.prompt_config.system_prompt`.

        """
        # If not provided, hydrate configuration from centralized settings
        if config is None:
            logger.debug("No config provided, loading from centralized settings")
            # Map string context_type from settings to local enum
            try:
                context_enum = ContextType(settings.context_type)
            except ValueError:
                logger.warning(
                    f"Invalid context_type '{settings.context_type}' in settings, "
                    "defaulting to FULL_DOCUMENT"
                )
                context_enum = ContextType.FULL_DOCUMENT

            config = DataExtractionConfig(
                model=settings.model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                context_type=context_enum,
                max_context_length=settings.max_context_length,
                selected_attribute_ids=list(settings.selected_attribute_ids),
                prompt_config=PromptConfig(),
                include_reasoning=settings.include_reasoning,
                include_additional_text=settings.include_additional_text,
            )
            logger.debug(
                f"Loaded config from settings: model={config.model}, "
                f"context_type={config.context_type}, "
                f"selected_attributes={len(config.selected_attribute_ids)}"
            )

        self.config = config
        self.custom_system_prompt_file = custom_system_prompt_file

        # Set up LLM model; prefer centralized settings, preserve env fallback
        azure_api_key_present = (
            settings.azure_api_key.get_secret_value()
            if settings.azure_api_key
            else None
        ) or os.getenv("AZURE_API_KEY")

        if azure_api_key_present:
            deployment = settings.azure_deployment or os.getenv(
                "AZURE_DEPLOYMENT", self.config.model
            )
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
            try:
                document_annotations = self.extract_from_document(document, attributes)
                all_annotations.extend(document_annotations)
            except (ValidationError, ValueError) as e:
                logger.error(
                    f"Failed to extract from document {document.document_id}: {e}"
                )
                logger.debug(f"Document: {document.name}")
                # Continue processing other documents
                continue

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

        Raises:
            ValueError: If no attributes are selected for extraction after filtering.

        """
        # Filter attributes based on selection mode
        selected_attributes = self._filter_attributes(attributes)

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        # Prepare context
        context = self._prepare_context(document)

        # Generate user message JSON payload
        prompt = self._generate_user_message_json(context, selected_attributes)

        # Call LLM
        llm_response = self._call_llm(prompt)

        # Parse response and create annotations
        return self._parse_llm_response(llm_response, selected_attributes, document)

    def _filter_attributes(
        self, attributes: list[EppiAttribute]
    ) -> list[EppiAttribute]:
        """Filter attributes using selected_attribute_ids if provided."""
        if self.config.selected_attribute_ids:
            filtered = [
                attr
                for attr in attributes
                if attr.attribute_id in self.config.selected_attribute_ids
            ]
            logger.debug(
                f"Filtered {len(attributes)} attributes to {len(filtered)} "
                f"using selected_attribute_ids: {self.config.selected_attribute_ids}"
            )
            return filtered
        logger.debug(
            f"No attribute filtering applied, using all {len(attributes)} attributes"
        )
        return attributes

    def _prepare_context(self, document: EppiDocument) -> str:
        """Prepare context based on context type."""
        if self.config.context_type == ContextType.FULL_DOCUMENT:
            context = document.context
            logger.debug(f"Using full document context (length: {len(str(context))})")
        elif self.config.context_type == ContextType.ABSTRACT_ONLY:
            context = document.abstract or document.context
            logger.debug(f"Using abstract-only context (length: {len(str(context))})")
        elif self.config.context_type == ContextType.RAG_SNIPPETS:
            # For now, use abstract as RAG snippets placeholder
            # In the future, this could be replaced with actual RAG snippets
            logger.warning(
                "RAG_SNIPPETS context type is using abstract as placeholder. "
                "This may not be the intended behavior. Actual RAG snippets "
                "should be implemented in the future."
            )
            context = document.abstract or document.context
            logger.debug(
                f"Using RAG snippets placeholder (abstract) "
                f"(length: {len(str(context))})"
            )
        elif self.config.context_type == ContextType.CUSTOM:
            context = document.context
            logger.debug(f"Using custom context (length: {len(str(context))})")
        else:
            msg = f"Unexpected context type: {self.config.context_type}"
            raise ValueError(msg)

        # Ensure context is a string
        if isinstance(context, list):
            logger.debug(f"Converting list context to string (items: {len(context)})")
            context = " ".join(context)

        # Truncate if too long
        original_length = len(context)
        if original_length > self.config.max_context_length:
            logger.debug(
                f"Truncating context from {original_length} to "
                f"{self.config.max_context_length} characters"
            )
            context = context[: self.config.max_context_length] + "..."

        return context

    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        # Use custom prompt file if provided
        if self.custom_system_prompt_file:
            prompt_file = self.custom_system_prompt_file
            logger.debug(f"Loading custom system prompt from {prompt_file}")
        else:
            prompt_file = (
                Path(__file__).parent.parent / "prompts" / "system_prompt_v0.txt"
            )
            logger.debug(f"Loading default system prompt from {prompt_file}")

        try:
            prompt_content = prompt_file.read_text(encoding="utf-8")
            logger.debug(f"Loaded system prompt ({len(prompt_content)} characters)")
            return prompt_content  # noqa: TRY300
        except FileNotFoundError:
            logger.warning(
                f"System prompt file not found at {prompt_file}, using default"
            )
            return self.config.prompt_config.system_prompt

    def _generate_user_message_json(
        self, context: str, attributes: list[EppiAttribute]
    ) -> str:
        """
        Generate structured JSON input for the LLM user message.

        The payload contains the prepared document context and an array of
        attributes shaped like `EppiAttribute`, where `question_target` is set
        from `attribute_set_description` by default. This function returns a
        JSON string to be used as the user message content.

        Args:
            context: Prepared document context string.
            attributes: List of EPPI attributes to evaluate.

        Returns:
            JSON string containing `context` and `attributes`.

        """
        logger.debug(f"Generating prompt for {len(attributes)} attributes")
        # Build attribute dictionaries ensuring keys align with EppiAttribute schema
        attributes_payload: list[dict[str, object]] = [
            {
                "attribute_id": attr.attribute_id,
                "attribute_label": attr.attribute_label,
                # EPPI uses boolean output
                "output_data_type": attr.output_data_type,
                # Use attribute_set_description as default question target
                "question_target": attr.attribute_set_description
                or "No description available",
                "attribute_set_description": attr.attribute_set_description,
            }
            for attr in attributes
        ]

        payload = {
            "context": context,
            "attributes": attributes_payload,
        }

        prompt_json = json.dumps(payload, ensure_ascii=False)
        logger.debug(f"Generated prompt JSON ({len(prompt_json)} characters)")
        return prompt_json

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        # Load system prompt from file
        system_prompt = self._load_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_annotation_response",
                    "schema": LLMResponseSchema.model_json_schema(),
                    "strict": True,
                },
            },
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
        """
        Parse and validate LLM response against EppiGoldStandardAnnotation structure.

        Args:
            response_content: Raw JSON string response from LLM
            attributes: List of attributes to match against
            document: Document being processed

        Returns:
            List of EppiGoldStandardAnnotation objects

        Raises:
            ValidationError: If response fails schema validation.
            ValueError: If JSON parsing fails.

        """
        try:
            # Validate against LLMResponseSchema
            validated_response = LLMResponseSchema.model_validate_json(response_content)
        except ValidationError as e:
            logger.error(f"LLM response failed schema validation: {e}")
            logger.debug(f"Response content: {response_content}")
            raise
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in LLM response: {e}"
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(error_msg) from e

        annotations = []
        logger.debug(
            f"Parsing LLM response with {len(validated_response.annotations)} "
            f"annotations"
        )
        for llm_annotation in validated_response.annotations:
            # Resolve attribute_id to full EppiAttribute
            attribute = next(
                (
                    attr
                    for attr in attributes
                    if attr.attribute_id == llm_annotation.attribute_id
                ),
                None,
            )

            if not attribute:
                logger.warning(
                    f"No attribute found for ID: {llm_annotation.attribute_id}"
                )
                continue

            # Convert to full EppiGoldStandardAnnotation
            annotation = EppiGoldStandardAnnotation(
                attribute=attribute,
                output_data=llm_annotation.output_data,
                annotation_type=AnnotationType.LLM,
                additional_text=llm_annotation.additional_text
                if self.config.include_additional_text
                else None,
                reasoning=llm_annotation.reasoning
                if self.config.include_reasoning
                else None,
            )
            annotations.append(annotation)
            logger.debug(
                f"Created annotation for attribute {attribute.attribute_id}: "
                f"output_data={llm_annotation.output_data}"
            )

        logger.debug(f"Successfully parsed {len(annotations)} annotations")
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
    """Extract a batch of specific attributes from a document by IDs."""
    if config is None:
        config = DataExtractionConfig()
    config.selected_attribute_ids = attribute_ids
    module = DataExtractionModule(config)
    return module.extract_from_document(document, attributes)
