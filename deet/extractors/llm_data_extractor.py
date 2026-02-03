"""Generalisable data extraction module for LLM-based document analysis."""

import json
from pathlib import Path

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    ContextType,
    GoldStandardAnnotation,
    LLMInputSchema,
    LLMResponseSchema,
)
from deet.logger import logger
from deet.settings import get_settings

settings = get_settings()


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    model_config = ConfigDict()

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

    model_config = ConfigDict()

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

    selected_attribute_ids: list[int] = Field(
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
        self,
        attributes: list[Attribute],
        payload: str,
        context_type: ContextType | None,
        prompt_outfile: Path | None = None,
    ) -> list[GoldStandardAnnotation]:
        """
        Extract data from a single document.

        Args:
            attributes: List of attributes to extract
            payload: str with the actual contents of the document to be analysed.
            context_type: Type of context to use (ContextType enum)
            prompt_outfile: path to write whole prompt to, optional.

        Returns:
            List of annotations for the document

        Raises:
            ValueError: If no attributes are selected for extraction after filtering.

        """
        # Convert config's selected_attribute_ids (list[str]) to list[int] for filtering
        filter_ids: list[int] | None = None
        if self.config.selected_attribute_ids:
            try:
                filter_ids = [
                    int(attr_id) for attr_id in self.config.selected_attribute_ids
                ]
            except (ValueError, TypeError):
                # If conversion fails, set to empty list so no attributes match
                logger.warning(
                    f"Invalid attribute IDs in config: "
                    f"{self.config.selected_attribute_ids}. "
                    "No attributes will be selected."
                )
                filter_ids = []

        selected_attributes = self._filter_attributes(attributes, filter_ids=filter_ids)

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(payload=payload, context_type=context_type)
        prompt = self._generate_user_message_json(
            payload=context, attributes=selected_attributes
        )
        llm_response = self._call_llm(prompt=prompt, prompt_outfile=prompt_outfile)

        return self._parse_llm_response(
            response_content=llm_response, attributes=selected_attributes
        )

    def extract_from_documents(
        self,
        payload: str,  # change to list[str] once we run for multiple pdfs.
        attributes: list[Attribute],
        output_file: Path | None = None,
        context_type: ContextType = ContextType.FULL_DOCUMENT,
        prompt_outfile: Path | None = None,
    ) -> list[GoldStandardAnnotation]:
        """
        Extract data from multiple documents.

        NOTE: placeholder, as we're still only working
        on one doc.

        Args:
            attributes: List of attributes to extract
            payload: the document(s) from which to extract data.
            output_file: Optional file to save parsed LLM response.
            context: a ContextType enum.
            prompt_outfile: optional path to write out complete prompt config.


        Returns:
            List of annotations for all documents and attributes

        """
        all_annotations = []
        document_annotations = self.extract_from_document(
            attributes,
            payload=payload,
            context_type=context_type,
            prompt_outfile=prompt_outfile,
        )
        all_annotations.extend(document_annotations)

        if output_file:
            self._save_results(all_annotations, output_file)

        return all_annotations

    def _filter_attributes(
        self, attributes: list[Attribute], filter_ids: list[int] | None = None
    ) -> list[Attribute]:
        """
        Filter attributes using provided attribute IDs.

        Args:
            attributes: List of attributes to filter
            filter_ids: Optional list of attribute IDs (ints) to filter by.
                        If None, returns all attributes.
                        If empty list, returns empty list.

        Returns:
            Filtered list of attributes matching the provided IDs, or all attributes
            if filter_ids is None, or empty list if filter_ids is empty.

        """
        if filter_ids is None:
            logger.debug(
                f"No attribute filtering applied, "
                f"using all {len(attributes)} attributes"
            )
            return attributes

        filtered = [attr for attr in attributes if attr.attribute_id in filter_ids]
        logger.debug(
            f"Filtered {len(attributes)} attributes to {len(filtered)} "
            f"using filter_ids: {filter_ids}"
        )
        return filtered

    def _prepare_context(  # type: ignore[mypy-note]
        # mypy incorrectly flags 'context' as possibly unbound;
        # all ContextType branches assign it before use.
        self,
        payload: str | list[str],  # NOTE: payload may be a single document (str)
        # or multiple documents/snippets (list[str]).
        context_type: ContextType | None = None,
    ) -> str:
        """Prepare context/payload based on context type."""
        if context_type is None:
            logger.debug("custom `context_type` not provided, using config-level.")
            context_type = self.config.context_type

        if context_type == ContextType.FULL_DOCUMENT:
            context = payload
            logger.debug(f"Using full document context (length: {len(str(context))})")
        # elif self.config.context_type == ContextType.ABSTRACT_ONLY:
        #     context = document.context  # type: ignore[assignment]
        #     logger.debug(f"Using abstract context (length: {len(str(context))})")
        elif context_type == ContextType.RAG_SNIPPETS:
            rag_not_impl = "rag-snippets context type is not implemented."
            raise NotImplementedError(rag_not_impl)
        elif context_type == ContextType.CUSTOM:
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
        payload: str,
        attributes: list[Attribute],
    ) -> str:
        """
        Generate structured JSON input for the LLM user message.

        The payload contains the prepared document context and an array
        `LLMInputSchema` objects, containing the attribute id, prompt and
        target output data type.

        NOTE: If `prompt` field is not populated in incoming data,
        LLMInputSchema will populate from `attribute_label`
        field, or fail.

        Args:
            context: Prepared document context string.
            attributes: List of Attribute objects to extract.

        Returns:
            JSON string containing `context` and `attributes`.

        """
        logger.debug(f"Generating prompt for {len(attributes)} attributes")
        attributes_payload = []
        for attr in attributes:
            # validate schema & fill prompt if not yet filled
            # Use exclude_none=False to ensure prompt field is included even if None
            attr_dict = attr.model_dump(exclude_none=False)
            llm_input_attr = LLMInputSchema(**attr_dict)
            attributes_payload.append(llm_input_attr.model_dump())

        unserialised_prompt = {
            "context": payload,
            "attributes": attributes_payload,
        }

        logger.debug(f"attributes payload: {attributes_payload}")
        prompt_json = json.dumps(unserialised_prompt, ensure_ascii=False)
        logger.debug(f"Generated prompt JSON ({len(prompt_json)} characters)")

        return prompt_json

    def _call_llm(self, prompt: str, prompt_outfile: Path | None = None) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt (str): the prompt
            prompt_outfile (Path | None, optional): a path to
            write the whole prompt (sys prompt + user message + context) as json.
            Defaults to None.

        Returns:
            str: the llm's response, to be parsed.

        """
        messages: list[dict] = [
            {"role": "system", "content": self.config.prompt_config.system_prompt},
            {"role": "user", "content": prompt},
        ]

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
        annotations_json = [x.model_dump() for x in annotations]
        output_file.write_text(json.dumps(annotations_json))
        logger.info(f"Results saved to: {output_file}")
