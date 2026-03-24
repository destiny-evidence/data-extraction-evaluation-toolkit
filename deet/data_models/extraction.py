"""Data models for LLM extraction outputs."""

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator

from deet.data_models.base import GoldStandardAnnotation
from deet.data_models.documents import GoldStandardAnnotatedDocument
from deet.utils.tokenisation import estimate_cost_usd, merge_prompt_completion_cost_usd


class DocumentExtractionResult(BaseModel):
    """Result of extracting data from a single document via an LLM."""

    annotations: list[GoldStandardAnnotation]
    messages: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    total_cost_usd: float | None = None
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
    per_document_tokens: dict[str, dict[str, int]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class ExtractionRunOutput(BaseModel):
    """Top-level output from a batch extraction run."""

    annotated_documents: list[GoldStandardAnnotatedDocument]
    metadata: ExtractionRunMetadata
