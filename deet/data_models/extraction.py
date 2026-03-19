"""Data models for LLM extraction outputs."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from deet.data_models.base import GoldStandardAnnotation
from deet.data_models.documents import GoldStandardAnnotatedDocument


class DocumentExtractionResult(BaseModel):
    """Result of extracting data from a single document via an LLM."""

    annotations: list[GoldStandardAnnotation]
    messages: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    total_cost_usd: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExtractionRunMetadata(BaseModel):
    """Aggregate metadata for a batch extraction run."""

    model: str | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float | None = None
    per_document_tokens: dict[str, dict[str, int]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExtractionRunOutput(BaseModel):
    """Top-level output from a batch extraction run."""

    annotated_documents: list[GoldStandardAnnotatedDocument]
    metadata: ExtractionRunMetadata
