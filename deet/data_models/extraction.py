"""Data models for LLM extraction outputs."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final, Self

from pydantic import BaseModel, Field

from deet.data_models.base import GoldStandardAnnotation
from deet.data_models.documents import GoldStandardAnnotatedDocument


class ExtractionPipelineStage(StrEnum):
    """Named stages timed during an extraction pipeline run."""

    annotation_conversion = "annotation_conversion"
    prompt_population = "prompt_population"
    document_preparation = "document_preparation"
    llm_extraction = "llm_extraction"
    artifact_export = "artifact_export"


PIPELINE_STAGE_NOTES: Final[dict[ExtractionPipelineStage, str]] = {
    ExtractionPipelineStage.annotation_conversion: (
        "Load and normalise the gold-standard import into documents and attributes."
    ),
    ExtractionPipelineStage.prompt_population: (
        "Attach custom prompts from the project prompt CSV. Zero if skipped."
    ),
    ExtractionPipelineStage.document_preparation: (
        "Load linked documents from cache or parse/read PDF and markdown sources. "
        "See per_document.parsing_seconds."
    ),
    ExtractionPipelineStage.llm_extraction: (
        "LLM extraction for each document. See per_document.llm_call_seconds."
    ),
    ExtractionPipelineStage.artifact_export: (
        "Write prompts snapshot, config snapshot, and this metadata file."
    ),
}


def pipeline_stage_notes() -> dict[str, str]:
    """Return human-readable notes keyed by pipeline stage name."""
    return {
        stage.value: PIPELINE_STAGE_NOTES[stage] for stage in ExtractionPipelineStage
    }


class RunMetadataNotes(BaseModel):
    """Human-readable notes for cost, token, and timing fields in run metadata."""

    total_pipeline_duration_seconds: str = (
        "Wall-clock time for the extraction run, from loading project data "
        "through export of experiment snapshots. Includes time spent answering "
        "the interactive config wizard when no config file is provided."
    )
    stage_durations_seconds: dict[str, str] = Field(
        default_factory=pipeline_stage_notes,
    )
    per_document: dict[str, str] = Field(
        default_factory=lambda: {
            "parsing_seconds": (
                "Parse/read time during document preparation; null when "
                "parsing_skipped is true."
            ),
            "parsing_skipped": (
                "True when full text came from cache and was not parsed this run."
            ),
            "llm_call_seconds": (
                "Wall-clock time for the LLM request, including local prompt setup."
            ),
            "cached_input_tokens": (
                "Prompt tokens served from the provider cache for this document. "
                "Already included in input_tokens. Zero when the provider reports "
                "no cache hit."
            ),
        },
    )
    total_cached_input_tokens: str = (
        "Sum of prompt tokens served from the provider cache across documents. "
        "Already included in total_input_tokens."
    )
    total_cost_usd: str = (
        "Sum of per-call costs reported by LiteLLM. None when no call could "
        "be priced. Reflects prompt-cache discounts and the endpoint that "
        "served each call."
    )


class DocumentParsingStats(BaseModel):
    """Parsing timing for a single document during document preparation."""

    parsing_seconds: float | None = None
    parsing_skipped: bool = True


class PerDocumentExtractionStats(BaseModel):
    """Per-document tokens, cached prompt tokens, and timing for an extraction run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    parsing_seconds: float | None = None
    parsing_skipped: bool = True
    llm_call_seconds: float = 0.0


class DocumentExtractionResult(BaseModel):
    """An LLM extraction result for one or more extraction calls for a document."""

    annotations: list[GoldStandardAnnotation]
    messages: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    model: str | None = None
    total_cost_usd: float | None = None
    llm_call_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @classmethod
    def combine(cls, parts: list[Self]) -> Self:
        """Fold per-call parts of one document's extraction into a single result."""
        annotations: list[GoldStandardAnnotation] = []
        messages: list[dict[str, Any]] = []
        models: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        llm_call_seconds = 0.0
        total_cost_usd: float | None = None

        for part in parts:
            annotations.extend(part.annotations)
            messages.extend(part.messages)
            input_tokens += part.input_tokens
            output_tokens += part.output_tokens
            cached_input_tokens += part.cached_input_tokens
            llm_call_seconds += part.llm_call_seconds
            if part.model is not None and part.model not in models:
                models.append(part.model)
            if part.total_cost_usd is not None:
                total_cost_usd = (total_cost_usd or 0.0) + part.total_cost_usd

        return cls(
            annotations=annotations,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            model=" + ".join(models) or None,
            llm_call_seconds=llm_call_seconds,
            total_cost_usd=total_cost_usd,
        )


class ExtractionRunMetadata(BaseModel):
    """Aggregate metadata for a batch extraction run."""

    model: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_cost_usd: float | None = None
    per_document: dict[str, PerDocumentExtractionStats] = Field(default_factory=dict)
    total_pipeline_duration_seconds: float | None = None
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    notes: RunMetadataNotes = Field(default_factory=RunMetadataNotes)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExtractionRunOutput(BaseModel):
    """Top-level output from a batch extraction run."""

    annotated_documents: list[GoldStandardAnnotatedDocument]
    metadata: ExtractionRunMetadata
