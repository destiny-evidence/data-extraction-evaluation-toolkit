"""Generalisable data extraction module for LLM-based document analysis."""

import json
from pathlib import Path
from typing import Any, cast

import litellm
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    GoldStandardAnnotation,
    LLMInputSchema,
    LLMResponseSchema,
)
from deet.data_models.documents import ContextType
from deet.logger import logger
from deet.settings import LLMProvider, get_settings
from deet.utils.tokenization import (
    count_tokens,
    get_model_max_tokens,
    truncate_to_token_limit,
)

settings = get_settings()


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    model_config = ConfigDict()

    system_prompt: str | Path = Field(
        description="System prompt that defines the task and role",
        default_factory=lambda: Path(__file__).parent.parent.parent
        / "prompts/system_prompt.txt",
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


def _model_string_for_tokenization() -> str:
    """Build the model string used for tokenization (matches LLMDataExtractor.model)."""
    if settings.llm_provider == LLMProvider.AZURE:
        return f"azure/{settings.azure_deployment}"
    if settings.llm_provider == LLMProvider.OLLAMA:
        return f"ollama/{settings.llm_model}"
    return settings.llm_model


class DataExtractionConfig(BaseModel):
    """Configuration for data extraction tasks."""

    model_config = ConfigDict()

    # LLM
    model: str = settings.llm_model
    provider: LLMProvider = settings.llm_provider
    temperature: float = settings.llm_temperature
    max_tokens: int | None = settings.llm_max_tokens

    # Context
    default_context_type: ContextType = Field(
        default=ContextType.FULL_DOCUMENT, description="Type of context to provide"
    )
    max_context_length: int | None = Field(
        default_factory=lambda: settings.llm_max_context_length,
        description=(
            "Maximum context length in tokens (total payload: system + attributes + "
            "context). None = infer from model. Override to manage costs."
        ),
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

    @model_validator(mode="after")
    def populate_max_context_length_from_model(self) -> "DataExtractionConfig":
        """Populate max_context_length from model when not set."""
        if self.max_context_length is not None:
            return self
        model_str = _model_string_for_tokenization()
        inferred = get_model_max_tokens(model_str)
        if inferred is not None:
            self.max_context_length = inferred
        else:
            self.max_context_length = 128_000
        return self


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
        if settings.llm_provider == LLMProvider.AZURE:
            self.model = f"azure/{settings.azure_deployment}"
            self.llm_api_key = settings.azure_api_key.get_secret_value()  # type: ignore[union-attr]
            self.api_base = settings.azure_api_base.get_secret_value()  # type: ignore[union-attr]
        elif settings.llm_provider == LLMProvider.OLLAMA:
            self.model = f"ollama/{settings.llm_model}"
            self.llm_api_key = None
            self.api_base = None
        else:
            error_message = f"Unsupported LLM provider: {settings.llm_provider}"
            raise ValueError(error_message)

        logger.info(f"Using {settings.llm_provider} with model: {self.model}")
        if self.config.max_tokens is not None:
            logger.info(f"max_tokens={self.config.max_tokens}")

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
        filter_attribute_ids: list[int] | None = None,
        *,
        payload: str | None = None,
        md_path: Path | None = None,
        context_type: ContextType | None = None,
    ) -> tuple[list[GoldStandardAnnotation], list[dict[str, Any]], int]:
        """
        Extract data from a single document.

        Call with either payload (document text) or md_path (path to markdown file).
        If md_path is provided, the file is read and used as the payload.
        Prompt payloads are not written here; the batch entry point
        extract_from_documents writes them to prompt_outfile when provided.

        Args:
            attributes: List of attributes to extract.
            payload: Document text to extract from. Required if md_path not set.
            md_path: Path to a markdown file to read as payload.
                Required if payload not set.
            context_type: Override config context type; if None, use config default.

        Returns:
            Tuple of (list of annotations for the document, messages sent to the LLM,
            output token count from the LLM response).

        Raises:
            ValueError: If no attributes are selected for extraction after filtering.
            ValueError: If neither payload nor md_path provided, or both provided.

        """
        if (payload is None and md_path is None) or (
            payload is not None and md_path is not None
        ):
            msg = "Exactly one of payload or md_path must be provided"
            raise ValueError(msg)
        if md_path is not None:
            if not md_path.exists():
                msg = f"Markdown file not found: {md_path}"
                raise FileNotFoundError(msg)
            payload = md_path.read_text(encoding="utf-8")
        payload = cast("str", payload)

        selected_attributes = attributes
        if filter_attribute_ids and len(filter_attribute_ids) > 0:
            try:
                selected_attributes = self._filter_attributes(
                    selected_attributes, filter_ids=filter_attribute_ids
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid attribute IDs in config: "
                    f"{filter_attribute_ids}. "
                    "No attributes will be selected."
                )

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(
            payload=payload,
            context_type=context_type,
        )
        prompt = self._generate_user_message_json(
            payload=context, attributes=selected_attributes
        )
        llm_response, messages, output_tokens = self._call_llm(prompt=prompt)
        annotations = self._parse_llm_response(
            response_content=llm_response, attributes=selected_attributes
        )
        return annotations, messages, output_tokens

    def extract_from_documents(  # noqa: PLR0913
        self,
        attributes: list[Attribute],
        markdown_dir: Path,
        filter_attribute_ids: list[int] | None = None,
        output_file: Path | None = None,
        context_type: ContextType = ContextType.FULL_DOCUMENT,
        prompt_outfile: Path | None = None,
    ) -> dict[str, list[GoldStandardAnnotation]]:
        """
        Extract data from all documents in a directory.

        Loops over files in markdown_dir. For each file, calls
        extract_from_document with md_path and context_type. Results are
        merged into one dict; optional combined JSON is written to output_file.

        Args:
            attributes: List of attributes to extract.
            markdown_dir: Directory of markdown files (required).
            output_file: Optional path to save combined results JSON.
            context_type: Override config context type for each document.
            prompt_outfile: Optional path to write a single JSON object:
                keys are document (PDF) paths, values are prompt payload (messages).


        Returns:
            Dictionary mapping file paths to lists of annotations.

        """
        if not markdown_dir.exists() or not markdown_dir.is_dir():
            msg = f"markdown_dir must be an existing directory: {markdown_dir}"
            raise ValueError(msg)

        all_results: dict[str, list[GoldStandardAnnotation]] = {}
        prompt_payloads: dict[str, Any] = {}
        per_document_output_tokens: dict[str, int] = {}

        input_files = [f for f in markdown_dir.iterdir() if f.suffix == ".md"]
        if not input_files:
            missing_files = f"no files in dir {markdown_dir}"
            raise ValueError(missing_files)

        for input_file in sorted(input_files):
            logger.info(f"Processing file: {input_file.name} ({input_file})")
            try:
                result, messages, output_tokens = self.extract_from_document(
                    attributes=attributes,
                    filter_attribute_ids=filter_attribute_ids,
                    md_path=input_file,
                    context_type=context_type,
                )

                all_results[input_file.name] = result
                prompt_payloads[input_file.name] = messages
                per_document_output_tokens[input_file.name] = output_tokens

            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to process {input_file}: {e}")
                logger.debug("Error details", exc_info=True)

        if output_file is not None:
            self._save_results(all_results, output_file, per_document_output_tokens)
            logger.info(f"Combined LLM classifications written to: {output_file}")

        if prompt_outfile is not None:
            prompt_outfile.write_text(
                json.dumps(prompt_payloads, indent=2), encoding="utf-8"
            )
            logger.info(f"Prompt payloads saved to: {prompt_outfile}")

        return all_results

    def _write_json_if_path(
        self, data: dict[str, Any] | list[Any], path: Path | None
    ) -> None:
        """
        Write data as JSON to path if path is not None; otherwise no-op.

        Args:
            data: Dict or list to serialize as JSON.
            path: Optional file path; when None, nothing is written.

        """
        if path is not None:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

    def _prepare_context(
        self,
        payload: str,
        context_type: ContextType | None = None,
    ) -> str:
        """Prepare context based on context type."""
        ctx = (
            context_type
            if context_type is not None
            else self.config.default_context_type
        )
        logger.debug(f"Using context type: {ctx}")
        if ctx == ContextType.FULL_DOCUMENT:
            context = payload
            logger.debug(f"Using full document context (length: {len(str(context))})")
        elif ctx == ContextType.RAG_SNIPPETS:
            rag_not_impl = "rag-snippets context type is not implemented."
            raise NotImplementedError(rag_not_impl)
        elif ctx == ContextType.CUSTOM:
            custom_not_impl = "custom context type is not implemented."
            raise NotImplementedError(custom_not_impl)
        else:
            other_not_allowed = f"{ctx} context type is not allowed."
            raise ValueError(other_not_allowed)

        if isinstance(context, list):
            logger.debug(f"Converting list context to string (items: {len(context)})")
            context = " ".join(context)

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
            payload: Prepared document context string.
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

    def _call_llm(self, prompt: str) -> tuple[str, list[dict[str, Any]], int]:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The user prompt (with context and attributes).

        Returns:
            Tuple of (LLM response text to be parsed, messages list sent to the API,
            output token count from the response).

        """
        system_prompt = self.config.prompt_config.system_prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Truncate context if total payload exceeds max_context_length
        try:
            max_ctx = self.config.max_context_length
            if max_ctx is None:
                pass  # No truncation when unset
            else:
                messages_text = " ".join(
                    str(m.get("content", "")) for m in messages
                )
                total_tokens = count_tokens(self.model, messages_text)
                if total_tokens > max_ctx:
                    prompt_data = json.loads(prompt)
                    context = prompt_data.get("context", "")
                    attributes_payload = prompt_data.get("attributes", [])
                    attributes_part = json.dumps(
                        {"context": "", "attributes": attributes_payload},
                        ensure_ascii=False,
                    )
                    system_tokens = count_tokens(self.model, str(system_prompt))
                    attributes_tokens = count_tokens(self.model, attributes_part)
                    # Buffer for token-count discrepancies or extra tokens from
                    # serialization/whitespace that LLM APIs may add.
                    buffer = 50
                    context_limit = max_ctx - system_tokens - attributes_tokens - buffer
                    if context_limit > 0:
                        context = truncate_to_token_limit(
                            context, self.model, context_limit
                        )
                        prompt_data["context"] = context
                        prompt = json.dumps(prompt_data, ensure_ascii=False)
                        messages[1]["content"] = prompt
                        logger.warning(
                            f"Truncated context to fit {max_ctx} "
                            "tokens. Edit `max_context_length` in your config."
                        )
                    else:
                        logger.warning(
                            "System prompt and attributes exceed "
                            "max_context_length; context will be empty."
                        )
                        prompt_data["context"] = ""
                        prompt = json.dumps(prompt_data, ensure_ascii=False)
                        messages[1]["content"] = prompt
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not truncate by tokens: {e}")

        logger.debug(f"Model: {self.model}")
        logger.debug(f"Temperature: {self.config.temperature}")
        logger.debug(f" sys message: {messages[0]['content'][:1000]}")
        logger.debug(f" user message: {messages[1]['content'][:1000]}")

        try:
            messages_text = " ".join(
                str(m.get("content", "")) for m in messages
            )
            input_tokens = count_tokens(self.model, messages_text)
            prompt_cost, _ = litellm.cost_per_token(
                model=self.model,
                prompt_tokens=input_tokens,
                completion_tokens=0,
            )
            logger.info(
                f"Estimated input cost: ${prompt_cost:.6f} USD ({input_tokens} tokens)"
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Could not compute input cost: {e}")

        response = litellm.completion(
            model=self.model,
            api_key=self.llm_api_key,
            api_base=self.api_base,
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

        output_tokens = (
            response.usage.completion_tokens
            if response.usage is not None
            and hasattr(response.usage, "completion_tokens")
            else 0
        )

        return response_content, messages, output_tokens

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
        self,
        results: dict[str, list[GoldStandardAnnotation]],
        output_file: Path,
        per_document_output_tokens: dict[str, int] | None = None,
    ) -> None:
        """
        Save results to file.

        Output uses nested structure: {results: {...}, metadata: {...}} to avoid
        mixing document keys with metadata and prevent filename collisions.

        Args:
            results: Dictionary mapping file paths to lists of annotations.
            output_file: Path to save results.
            per_document_output_tokens: Optional mapping of document name to
                output token count. When provided, adds metadata to the output.

        """
        results_data: dict[str, list[dict[str, Any]]] = {
            file_path: [ann.model_dump() for ann in annotations]
            for file_path, annotations in results.items()
        }
        output: dict[str, Any] = {"results": results_data}
        if per_document_output_tokens is not None:
            total_output_tokens = sum(per_document_output_tokens.values())
            output["metadata"] = {
                "total_output_tokens": total_output_tokens,
                "per_document": per_document_output_tokens,
            }
        output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info(f"Results saved to: {output_file}")
