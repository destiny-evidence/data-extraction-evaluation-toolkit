"""Base data extraction module to be extended with different extraction methods."""

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum, auto
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from deet.data_models.base import Attribute
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
)
from deet.ui.terminal.render import optional_progress
from deet.utils.tokenisation import (
    get_model_max_tokens,
)


def default_system_prompt() -> str:
    """Get default system prompt included in the package."""
    return (files("deet.prompts") / "system_prompt.txt").read_text()


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


class ExtractionMethod(StrEnum):
    """Enum of extraction methods."""

    LLM = auto()
    KEYWORD = auto()
    SEMANTIC = auto()


class DataExtractionConfig(BaseModel):
    """Configuration for data extraction tasks."""

    model_config = ConfigDict()

    method: ExtractionMethod = Field(
        default=ExtractionMethod.LLM, description="Extraction Method"
    )

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


class BaseDataExtractor(ABC):
    """Abstract Base Class defining common methods for data extractors."""

    def __init__(self, config: DataExtractionConfig) -> None:
        """Initialise with data extraction config."""
        self.config = config

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

        annotated_docs: list[GoldStandardAnnotatedDocument] = []
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

                    annotated_docs.append(
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
            model=self.config.model,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
            per_document_tokens=per_doc_tokens,
        )
        run_output = ExtractionRunOutput(
            annotated_documents=annotated_docs,
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

    @abstractmethod
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
