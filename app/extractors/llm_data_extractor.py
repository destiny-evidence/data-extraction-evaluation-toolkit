"""Generalisable data extraction module for LLM-based document analysis."""

import json
from enum import StrEnum, auto
from pathlib import Path

import litellm
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.data_models.base import (
    AnnotationType,
    Attribute,
    Document,
    GoldStandardAnnotation,
    LLMInputSchema,
    LLMResponseSchema,
)
from app.logger import logger
from app.settings import get_settings

settings = get_settings()


class ContextType(StrEnum):
    """Types of context that can be provided to the LLM."""

    FULL_DOCUMENT = auto()
    ABSTRACT_ONLY = auto()
    RAG_SNIPPETS = auto()
    CUSTOM = auto()


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    system_prompt: str | Path = Field(
        description="System prompt that defines the task and role",
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "prompts/system_prompt.txt",
    )
    attribute_specific_prompt: str = Field(
        description="Prompt template for attribute-specific extraction",
        default=(
            "Analyse this research document and answer questions about "
            "specific attributes. For each attribute, determine if it is "
            "present (True/False), provide reasoning, and include citations."
        ),
    )
    single_attribute_prompt: str = Field(
        description="Prompt template for single attribute extraction",
        default=(
            "Analyse this research document for the following attribute: "
            "{attribute_label}. Determine if it is present (True/False), "
            "provide reasoning, and include citations."
        ),
    )
    batch_attribute_prompt: str = Field(
        description="Prompt template for batch attribute extraction",
        default=(
            "Analyse this research document and answer questions about the "
            "following attributes. For each attribute, determine if it's "
            "present (True/False), provide reasoning, and include citations."
        ),
    )

    @model_validator(mode="after")
    def load_system_prompt_file(self) -> "PromptConfig":
        """Load system prompt from file if Path provided."""
        if isinstance(self.system_prompt, Path):
            if not self.system_prompt.exists():
                sys_prompt_missing = f"sys prompt {self.system_prompt} not found."
                raise ValueError(sys_prompt_missing)
            logger.debug(f"Reading system prompt from {self.system_prompt}")
            self.system_prompt = self.system_prompt.read_text()

        return self


class DataExtractionConfig(BaseModel):
    """Configuration for data extraction tasks."""

    # LLM
    model: str = settings.llm_model
    temperature: float = settings.llm_temperature
    max_tokens: int | None = settings.llm_max_tokens

    # Context
    context_type: ContextType = Field(
        default=ContextType.FULL_DOCUMENT, description="Type of context to provide"
    )
    max_context_length: int = Field(
        default=200000,
        description="Maximum context length for LLM CHARACTERS? ",  # fix with tokens!
    )

    selected_attribute_ids: list[str] = Field(
        default=[], description="Specific attribute IDs to extract"
    )

    # Prompt
    prompt_config: PromptConfig = Field(
        default_factory=PromptConfig, description="Prompt configuration"
    )

    # Output
    include_reasoning: bool = Field(
        default=True, description="Include reasoning in output"
    )
    include_additional_text: bool = Field(
        default=True, description="Include additional text/citations in output"
    )


class LLMDataExtractor:
    """
    Generalisable module for LLM-based data extraction from documents.

    This module provides a flexible interface for extracting structured data
    from documents using LLMs, with support for different context types and
    customizable prompts.
    """

    def __init__(
        self,
        config: DataExtractionConfig,
        custom_system_prompt_file: Path | None = None,
        *,
        show_litellm_debug_messages: bool = False,
    ) -> None:
        """
        Initialise the data extraction module.

        Args:
            config (DataExtractionConfig): config obj for data extraction run
            custom_system_prompt_file (Path | None, optional): path to non-defualt
            sys prompt file. Defaults to None.
            show_litellm_debug_messages (bool, optional): show verbose litellm logs.
            Defaults to False.

        """
        self.config = config
        self.custom_system_prompt_file = custom_system_prompt_file
        self.model = f"azure/{settings.azure_deployment}"
        self.azure_key = settings.azure_api_key.get_secret_value()  # type: ignore[union-attr]
        self.azure_base = settings.azure_api_base.get_secret_value()  # type: ignore[union-attr]
        if show_litellm_debug_messages:
            litellm._turn_on_debug()  # noqa: SLF001

        if (
            self.custom_system_prompt_file
            and self.custom_system_prompt_file
            != self.config.prompt_config.system_prompt
        ):
            logger.debug("found custom sys prompt. loading...")
            self.config.prompt_config.system_prompt = (
                self.custom_system_prompt_file.read_text()
            )

    def extract_from_document(
        self, document: Document, attributes: list[Attribute], **kwargs
    ) -> list[GoldStandardAnnotation]:
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
        selected_attributes = self._filter_attributes(attributes)

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(document, **kwargs)
        prompt = self._generate_user_message_json(context, selected_attributes)
        llm_response = self._call_llm(prompt, **kwargs)

        return self._parse_llm_response(llm_response, selected_attributes)

    def extract_from_documents(
        self,
        documents: list[Document],
        attributes: list[Attribute],
        output_file: Path | None = None,
        **kwargs,
    ) -> list[GoldStandardAnnotation]:
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
                document_annotations = self.extract_from_document(
                    document, attributes, **kwargs
                )
                all_annotations.extend(document_annotations)
            except (ValidationError, ValueError) as e:
                logger.error(
                    f"Failed to extract from document {document.document_id}: {e}"
                )
                logger.debug(f"Document: {document.name}")
                continue

        if output_file:
            self._save_results(all_annotations, output_file)

        return all_annotations

    def _filter_attributes(self, attributes: list[Attribute]) -> list[Attribute]:
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

    def _prepare_context(self, document: Document, full_text: str, **kwargs) -> str:
        """Prepare context based on context type."""
        if self.config.context_type == ContextType.FULL_DOCUMENT:
            context = full_text
            logger.debug(f"Using full document context (length: {len(str(context))})")
        elif self.config.context_type == ContextType.ABSTRACT_ONLY:
            context = document.context  # type: ignore[assignment]
            logger.debug(f"Using abstract context (length: {len(str(context))})")
        elif self.config.context_type == ContextType.RAG_SNIPPETS:
            rag_not_impl = "rag-snippets context type is not implemented."
            raise NotImplementedError(rag_not_impl)
        elif self.config.context_type == ContextType.CUSTOM:
            custom_not_impl = "custom context type is not implemented."
            raise NotImplementedError(custom_not_impl)
        else:
            other_not_allowed = (
                f"{self.config.context_type} context type is not allowed."
            )
            raise ValueError(other_not_allowed)

        if isinstance(context, list):
            logger.debug(f"Converting list context to string (items: {len(context)})")
            context = " ".join(context)

        # Truncate if too long
        original_length = len(context)
        if original_length > self.config.max_context_length:
            logger.warning(
                f"Truncating context from {original_length} to "
                f"{self.config.max_context_length} characters. "
                "If you want to change the number of characters that get truncated, "
                "edit `max_context_length` in your config."
            )
            context = context[: self.config.max_context_length] + "..."

        return context

    def _generate_user_message_json(
        self,
        context: str,
        attributes: list[Attribute],
    ) -> str:
        """
        Generate structured JSON input for the LLM user message.

        The payload contains the prepared document context and an array
        `LLMInputSchema` objects, containing the attribute id, prompt and
        target output data type.

        NOTE: If `prompt` field is not populated in incoming data,
        LLMInputSchema will populate from `attribute_set_description`
        field, or fail.

        Args:
            context: Prepared document context string.
            attributes: List of LLMInputSchema.

        Returns:
            JSON string containing `context` and `attributes`.

        """
        logger.debug(f"Generating prompt for {len(attributes)} attributes")
        attributes_payload = []
        for attr in attributes:
            # validate schema & fill prompt if not yet filled
            llm_input_attr = LLMInputSchema(**attr.model_dump())
            attributes_payload.append(llm_input_attr.model_dump())

        payload = {
            "context": context,
            "attributes": attributes_payload,
        }

        logger.debug(f"attributes payload: {attributes_payload}")
        prompt_json = json.dumps(payload, ensure_ascii=False)
        logger.debug(f"Generated prompt JSON ({len(prompt_json)} characters)")

        return prompt_json

    def _call_llm(
        self, prompt: str, prompt_outfile: Path | None = None, **kwargs
    ) -> str:
        """Call the LLM with the given prompt."""
        messages: list[dict] = [
            {"role": "system", "content": self.config.prompt_config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Path("misc/system_user_config.json").write_text(json.dumps(messages))

        logger.debug(f"Model: {self.model}")
        logger.debug(f"Temperature: {self.config.temperature}")
        logger.debug(f" sys message: {messages[0]['content'][:1000]}")
        logger.debug(f"user msg{messages[1]['content'][:1000]}")

        if prompt_outfile:
            prompt_outfile.write_text(json.dumps(messages))

        response = litellm.completion(
            model=self.model,
            api_key=self.azure_key,
            api_base=self.azure_base,
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
        logger.debug(f"raw response: {response_content}")

        return response_content

    def _parse_llm_response(
        self,
        response_content: str,
        attributes: list[Attribute],
    ) -> list[GoldStandardAnnotation]:
        """
        Parse and validate LLM response against GoldStandardAnnotation structure.

        Args:
            response_content: Raw JSON string response from LLM
            attributes: List of attributes to match against
            document: Document being processed

        Returns:
            List of GoldStandardAnnotation objects

        Raises:
            ValidationError: If response fails schema validation.
            ValueError: If JSON parsing fails.

        """
        try:
            validated_response = LLMResponseSchema.model_validate_json(response_content)
        except ValidationError as ve:
            logger.error(f"LLM response failed schema validation: {ve}")
            logger.debug(f"Response content: {response_content}")
            raise
        except json.JSONDecodeError as je:
            error_msg = f"Invalid JSON in LLM response: {je}"
            logger.error(f"Failed to parse LLM response as JSON: {je}")
            raise ValueError(error_msg) from je

        annotations = []
        logger.debug(
            f"Parsing LLM response with {len(validated_response.annotations)} "
            f"annotations"
        )
        for llm_annotation in validated_response.annotations:
            # Resolve attribute_id to full Attribute
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

            additional_text = (
                llm_annotation.additional_text
                if self.config.include_additional_text
                else None
            )
            reasoning = (
                llm_annotation.reasoning if self.config.include_reasoning else None
            )
            # Convert to full EppiGoldStandardAnnotation
            annotation = GoldStandardAnnotation(
                attribute=attribute,
                output_data=llm_annotation.output_data,
                annotation_type=AnnotationType.LLM,
                additional_text=additional_text,
                reasoning=reasoning,
            )
            annotations.append(annotation)
            logger.debug(
                f"Created annotation for attribute {attribute.attribute_id}: "
                f"output_data={llm_annotation.output_data}"
            )

        logger.debug(f"Successfully parsed {len(annotations)} annotations")
        return annotations

    def _save_results(
        self, annotations: list[GoldStandardAnnotation], output_file: Path
    ) -> None:
        """Save results to file."""
        annotations_data = ""
        for annotation in annotations:
            annotation_json_str = annotation.model_dump_json()
            annotations_data += annotation_json_str
            annotations_data += "\n"

        output_file.write_text(annotations_data)

        logger.info(f"Results saved to: {output_file}")
