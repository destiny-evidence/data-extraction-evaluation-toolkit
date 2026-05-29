"""Generalisable data extraction module for LLM-based document analysis."""

import json
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, cast

import litellm
import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    GoldStandardAnnotation,
    LLMInputSchema,
    LLMResponseSchema,
)
from deet.data_models.documents import (
    ContextType,
    Document,
    GoldStandardAnnotatedDocument,
)
from deet.data_models.extraction import (
    DocumentExtractionResult,
    ExtractionRunMetadata,
    ExtractionRunOutput,
)
from deet.data_models.ui_schema import UI
from deet.exceptions import LitellmModelNotMappedError
from deet.settings import (
    DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK,
    LLMProvider,
    get_settings,
)
from deet.ui.terminal.render import optional_progress
from deet.utils.tokenisation import (
    count_tokens,
    estimate_cost_usd,
    get_model_max_tokens,
    truncate_to_token_limit,
)

settings = get_settings()


def default_system_prompt() -> str:
    """Get default system prompt included in the package."""
    return (files("deet.prompts") / "system_prompt.txt").read_text()


class PromptConfig(BaseModel):
    """Configuration for prompts used in data extraction."""

    model_config = ConfigDict()

    system_prompt: str | Path = Field(
        description="System prompt that defines the task and role",
        default_factory=default_system_prompt,
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


def _model_string_for_tokenisation(provider: LLMProvider, model: str) -> str:
    """
    Build the model string used for tokenisation.

    Must match how ``LLMDataExtractor`` sets ``self.model`` from config.
    """
    match provider:
        case LLMProvider.AZURE:
            return f"azure/{model}"
        case LLMProvider.OLLAMA:
            return f"ollama/{model}"
        case _:
            msg = f"Unsupported LLM provider: {provider}"
            raise ValueError(msg)


class DataExtractionConfig(BaseModel):
    """Configuration for data extraction tasks."""

    model_config = ConfigDict()

    # LLM
    provider: Annotated[
        LLMProvider, UI(help="Choose from a list of supported LLM providers.")
    ] = Field(default=LLMProvider.AZURE, description="LLM Provider")
    model: Annotated[str, UI(help="The name of the LLM model you want to use.")] = (
        Field(
            default="gpt-4o-mini",
            description="LLM model identifier used for completions.",
        )
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for the LLM.",
        ge=0.0,
    )
    max_tokens: Annotated[
        int | None,
        UI(
            help=(
                "The maximum number of tokens in the LLM response. "
                "Leave blank for the provider default."
            )
        ),
    ] = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (Leave blank for provider default)."
        ),
    )

    max_context_tokens: Annotated[
        int | None,
        UI(
            help=("Maximum input context length " "(Leave blank for provider default).")
        ),
    ] = Field(
        default=None,
        description=(
            "Maximum input context length in tokens (system + attributes + "
            "document). None = infer from model (litellm registry), else "
            f"{DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK} via "
            "DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK. Override to manage costs."
        ),
    )

    # Context
    default_context_type: Annotated[
        ContextType, UI(help="Where to extract data from.")
    ] = Field(
        default=ContextType.FULL_DOCUMENT, description="Type of context to provide"
    )

    truncate_on_overflow: Annotated[
        bool,
        UI(
            help=(
                "Select true to truncate documents longer than max_context_tokens. "
                "This will ensure extraction runs without crashing, but may mean"
                " some parts of the document are not seen by the LLM."
            )
        ),
    ] = Field(
        default=False,
        description=(
            "When True, automatically truncate context that exceeds "
            "max_context_tokens. When False (default), raise ValueError."
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
    def populate_max_context_tokens_from_model(self) -> "DataExtractionConfig":
        """Populate max_context_tokens from model when not set."""
        if self.max_context_tokens is not None:
            return self
        model_str = _model_string_for_tokenisation(self.provider, self.model)
        try:
            inferred = get_model_max_tokens(model_str)
        except LitellmModelNotMappedError:
            inferred = None
        if inferred is not None:
            self.max_context_tokens = inferred
        else:
            # Use shared fallback when model max tokens cannot be inferred.
            self.max_context_tokens = DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "DataExtractionConfig":
        """Load config object from a yaml file."""
        if not path.exists():
            not_found = f"Config file not found at: {path}"
            raise FileNotFoundError(not_found)

        return cls.model_validate(yaml.safe_load(path.read_text()))


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
        if self.config.provider == LLMProvider.AZURE:
            self.model = f"azure/{self.config.model}"
            self.llm_api_key = settings.azure_api_key.get_secret_value()  # type: ignore[union-attr]
            self.api_base = settings.azure_api_base.get_secret_value()  # type: ignore[union-attr]
        elif self.config.provider == LLMProvider.OLLAMA:
            self.model = f"ollama/{self.config.model}"
            self.llm_api_key = None
            self.api_base = None
        else:
            error_message = f"Unsupported LLM provider: {self.config.provider}"
            raise ValueError(error_message)

        logger.info(f"Using {self.config.provider} with model: {self.model}")
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
    ) -> DocumentExtractionResult:
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
            DocumentExtractionResult with annotations, messages, token counts,
            cost, model name, and timestamp.

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

        context = self._prepare_context(payload=payload, context_type=context_type)
        prompt = self._generate_user_message_json(
            payload=context, attributes=selected_attributes
        )
        llm_response, messages, output_tokens, input_tokens = self._call_llm(
            prompt=prompt
        )
        annotations = self._parse_llm_response(
            response_content=llm_response, attributes=selected_attributes
        )

        return DocumentExtractionResult(
            annotations=annotations,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    def extract_from_documents(  # noqa: PLR0913
        self,
        attributes: list[Attribute],
        documents: Sequence[Document],
        filter_attribute_ids: list[int] | None = None,
        output_file: Path | None = None,
        context_type: ContextType | None = None,
        prompt_outfile: Path | None = None,
        *,
        show_progress: bool = False,
    ) -> ExtractionRunOutput:
        """
        Extract data from all documents.

        Loops over documents and extracts data using list of attributes.

        Args:
            attributes: List of attributes to extract.
            documents: Sequence of Document instances (required).
            filter_attribute_ids: Optional list of attribute IDs to filter by.
            output_file: Optional path to save combined results JSON.
            context_type: Override config context type; if None, use config default.
            prompt_outfile: Optional path to write a single JSON object:
                keys are document IDs, values are prompt payload (messages).
            show_progress: Whether to show a progress bar.

        Returns:
            ExtractionRunOutput containing annotated documents and run metadata.

        """
        if context_type is None:
            context_type = self.config.default_context_type

        prompt_payloads: dict[str, Any] = {}
        per_doc_tokens: dict[str, dict[str, int]] = {}

        llm_annotated_docs: list[GoldStandardAnnotatedDocument] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost: float | None = None

        with optional_progress(
            documents, show_progress=show_progress
        ) as iterable_documents:
            for document in iterable_documents:
                logger.info(f"Processing document: {document.name}")

                if context_type == ContextType.ABSTRACT_ONLY:
                    document.set_abstract_context()
                elif context_type == ContextType.FULL_DOCUMENT:
                    document.context = document.safe_parsed_document.text

                try:
                    result = self.extract_from_document(
                        attributes=attributes,
                        filter_attribute_ids=filter_attribute_ids,
                        payload=document.context,
                        context_type=context_type,
                    )

                    llm_annotated_docs.append(
                        GoldStandardAnnotatedDocument(
                            document=document, annotations=result.annotations
                        )
                    )
                    doc_id_str = str(document.safe_identity.document_id)
                    prompt_payloads[doc_id_str] = result.messages
                    per_doc_tokens[doc_id_str] = {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                    total_input_tokens += result.input_tokens
                    total_output_tokens += result.output_tokens
                    if result.total_cost_usd is not None:
                        total_cost = (total_cost or 0.0) + result.total_cost_usd

                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to process {document.name}: {e}")
                    logger.debug("Error details", exc_info=True)

        run_metadata = ExtractionRunMetadata(
            model=self.model,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
            per_document_tokens=per_doc_tokens,
        )
        run_output = ExtractionRunOutput(
            annotated_documents=llm_annotated_docs,
            metadata=run_metadata,
        )

        if output_file is not None:
            self._save_results(run_output, output_file)
            logger.info(f"Combined LLM classifications written to: {output_file}")

        if prompt_outfile is not None:
            prompt_outfile.write_text(
                json.dumps(prompt_payloads, indent=2), encoding="utf-8"
            )
            logger.info(f"Prompt payloads saved to: {prompt_outfile}")

        return run_output

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
        elif ctx == ContextType.ABSTRACT_ONLY:
            context = payload
            logger.debug(f"Using abstract context (length: {len(str(context))})")
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

    def _enforce_context_limit(
        self,
        messages: list[dict[str, Any]],
        prompt: str,
        system_prompt: str,
    ) -> None:
        """
        Enforce max_context_tokens on the messages payload.

        When the total token count exceeds max_context_tokens:
        - If ``truncate_on_overflow`` is False (default), raises ValueError.
        - If ``truncate_on_overflow`` is True, truncates the "context" field
          inside the user prompt JSON to fit. Mutates messages in place.

        Args:
            messages: List of message dicts with "role" and "content";
                messages[1] is the user message (prompt JSON).
            prompt: Current user prompt JSON string.
            system_prompt: System prompt text (for token counting).

        Raises:
            ValueError: When payload exceeds max_context_tokens and
                truncate_on_overflow is False, or when truncate_on_overflow is
                True but the prompt JSON cannot be parsed or messages are not
                in the expected shape so truncation cannot be applied.

        """
        max_ctx = self.config.max_context_tokens
        if max_ctx is None:
            return
        messages_text = " ".join(str(m.get("content", "")) for m in messages)
        total_tokens = count_tokens(self.model, messages_text)
        if total_tokens <= max_ctx:
            return

        if not self.config.truncate_on_overflow:
            msg = (
                f"Payload ({total_tokens} tokens) exceeds "
                f"max_context_tokens ({max_ctx} tokens). "
                "Set truncate_on_overflow=True in your config to "
                "automatically truncate, or increase max_context_tokens."
            )
            raise ValueError(msg)

        try:
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
                context = truncate_to_token_limit(context, self.model, context_limit)
                prompt_data["context"] = context
                truncated_prompt = json.dumps(prompt_data, ensure_ascii=False)
                messages[1]["content"] = truncated_prompt
                logger.warning(
                    f"Truncated context to fit {max_ctx} "
                    "tokens. Edit `max_context_tokens` in your config."
                )
            else:
                logger.warning(
                    "System prompt and attributes exceed "
                    "max_context_tokens; context will be empty."
                )
                prompt_data["context"] = ""
                messages[1]["content"] = json.dumps(prompt_data, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug(f"Could not truncate by tokens: {e}")
            logger.warning(
                "Could not enforce max_context_tokens by truncating the prompt JSON; "
                "prompt appears to be invalid or not in the expected shape."
            )
            msg = (
                "Failed to enforce max_context_tokens: invalid or unexpected prompt "
                "JSON structure while truncate_on_overflow=True."
            )
            raise ValueError(msg) from e

    def _call_llm(self, prompt: str) -> tuple[str, list[dict[str, Any]], int, int]:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The user prompt (with context and attributes).

        Returns:
            Tuple of (LLM response text, messages list, output token count,
            input/prompt token count from the response usage).

        """
        system_prompt = self.config.prompt_config.system_prompt
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        self._enforce_context_limit(messages, prompt, str(system_prompt))

        logger.debug(f"Model: {self.model}")
        logger.debug(f"Temperature: {self.config.temperature}")
        logger.debug(f" sys message: {messages[0]['content'][:1000]}")
        logger.debug(f" user message: {messages[1]['content'][:1000]}")

        messages_text = " ".join(str(m.get("content", "")) for m in messages)
        input_tokens = count_tokens(self.model, messages_text)
        prompt_cost, _ = estimate_cost_usd(
            self.model,
            prompt_tokens=input_tokens,
            completion_tokens=0,
        )
        if prompt_cost is not None:
            logger.info(
                f"Estimated input cost: ${prompt_cost:.6f} USD "
                f"({input_tokens} tokens)"
            )

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

        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            if hasattr(response.usage, "prompt_tokens"):
                input_tokens = response.usage.prompt_tokens or 0
            if hasattr(response.usage, "completion_tokens"):
                output_tokens = response.usage.completion_tokens or 0

        return response_content, messages, output_tokens, input_tokens

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
                raw_data=llm_annotation.output_data,
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
        run_output: ExtractionRunOutput,
        output_file: Path,
    ) -> None:
        """
        Serialize an ExtractionRunOutput to JSON and write it to disk.

        Args:
            run_output: The complete extraction run output to persist.
            output_file: Destination file path.

        """
        output_file.write_text(
            run_output.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(f"Results saved to: {output_file}")
