"""Data models for LLM extraction outputs."""

from datetime import UTC, datetime
from typing import Any, Final, Self

from pydantic import BaseModel, Field, model_validator

from deet.data_models.base import GoldStandardAnnotation
from deet.data_models.documents import GoldStandardAnnotatedDocument
from deet.utils.tokenisation import estimate_cost_usd, merge_prompt_completion_cost_usd

RUN_STAGE_NOTES: Final[dict[str, str]] = {
    "annotation_conversion": (
        "Load and normalise the gold-standard import into documents and attributes."
    ),
    "prompt_population": (
        "Attach custom prompts from the project prompt CSV. Zero if skipped."
    ),
    "document_preparation": (
        "Load linked documents from cache or parse/read PDF and markdown sources. "
        "See per_document.parsing_seconds."
    ),
    "llm_extraction": (
        "LLM extraction for each document. See per_document.llm_call_seconds."
    ),
    "artifact_export": (
        "Write prompts snapshot, config snapshot, and this metadata file."
    ),
}


class RunMetadataNotes(BaseModel):
    """Human-readable notes for timing fields in run metadata."""

    total_pipeline_duration_seconds: str = (
        "Wall-clock time for the extraction run, from loading project data "
        "through export of experiment snapshots. Includes time spent in the "
        "interactive CLI config wizard when no config file is provided."
    )
    stage_durations_seconds: dict[str, str] = Field(
        default_factory=lambda: dict(RUN_STAGE_NOTES),
    )
    per_document: dict[str, str] = Field(
        default_factory=lambda: {
            "parsing_seconds": (
                "Parse/read time during document preparation; 0 when parsing_skipped "
                "is true."
            ),
            "parsing_skipped": (
                "True when full text came from cache and was not parsed this run."
            ),
            "llm_call_seconds": (
                "Wall-clock time for the LLM request, including local prompt setup."
            ),
        },
    )


class DocumentParsingStats(BaseModel):
    """Parsing timing for a single document during document preparation."""

    parsing_seconds: float = 0.0
    parsing_skipped: bool = True


class PerDocumentStats(BaseModel):
    """Per-document tokens and timing for an extraction run."""

    input_tokens: int = 0
    output_tokens: int = 0
    parsing_seconds: float = 0.0
    parsing_skipped: bool = True
    llm_call_seconds: float = 0.0


class DocumentExtractionResult(BaseModel):
    """Result of extracting data from a single document via an LLM."""

    annotations: list[GoldStandardAnnotation]
    messages: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    total_cost_usd: float | None = None
    llm_call_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @model_validator(mode="after")
    def compute_total_cost_usd(self) -> Self:
        """Populate ``total_cost_usd`` from tokens and model (``estimate_cost_usd``)."""
        if self.model is None:
            self.total_cost_usd = None
            return self
        prompt_c, completion_c = estimate_cost_usd(
            self.model,
            prompt_tokens=self.input_tokens,
            completion_tokens=self.output_tokens,
        )
        merged = merge_prompt_completion_cost_usd(prompt_c, completion_c)
        self.total_cost_usd = round(merged, 6) if merged is not None else None
        return self


class ExtractionRunMetadata(BaseModel):
    """Aggregate metadata for a batch extraction run."""

    model: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    per_document: dict[str, PerDocumentStats] = Field(default_factory=dict)
    total_pipeline_duration_seconds: float | None = None
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    notes: RunMetadataNotes = Field(default_factory=RunMetadataNotes)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExtractionRunOutput(BaseModel):
    """Top-level output from a batch extraction run."""

    annotated_documents: list[GoldStandardAnnotatedDocument]
    metadata: ExtractionRunMetadata
