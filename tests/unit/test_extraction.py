"""Tests for LLM extraction output data models."""

from collections.abc import Sequence

import pytest
from destiny_sdk.references import ReferenceFileInput

from deet.data_models.base import (
    AnnotationType,
    Attribute,
    AttributeType,
    GoldStandardAnnotation,
)
from deet.data_models.documents import (
    ContextType,
    Document,
    GoldStandardAnnotatedDocument,
)
from deet.data_models.extraction import (
    PIPELINE_STAGE_NOTES,
    DocumentExtractionResult,
    ExtractionPipelineStage,
    ExtractionRunMetadata,
    ExtractionRunOutput,
    PerDocumentExtractionStats,
    pipeline_stage_notes,
)


def _sample_attribute(attribute_id: int = 1) -> Attribute:
    """Build a minimal valid Attribute for tests."""
    return Attribute(
        prompt="Q?",
        output_data_type=AttributeType.BOOL,
        attribute_id=attribute_id,
        attribute_label=f"Attr {attribute_id}",
    )


def _sample_annotation(attribute_id: int = 1) -> GoldStandardAnnotation:
    """Build a minimal GoldStandardAnnotation with a distinct attribute id."""
    return GoldStandardAnnotation(
        attribute=_sample_attribute(attribute_id),
        raw_data=True,
        annotation_type=AnnotationType.LLM,
    )


def _sample_document() -> Document:
    """Build a minimal Document for GoldStandardAnnotatedDocument."""
    return Document(
        name="doc.pdf",
        citation=ReferenceFileInput(),
        context="ctx",
        context_type=ContextType.FULL_DOCUMENT,
        document_id=1,
    )


def _sample_gold_annotated_document() -> GoldStandardAnnotatedDocument:
    """Build a minimal GoldStandardAnnotatedDocument."""
    return GoldStandardAnnotatedDocument(
        document=_sample_document(),
        annotations=[_sample_annotation()],
    )


def test_document_extraction_result_defaults() -> None:
    """Empty annotations/messages and no model yield no cost."""
    result = DocumentExtractionResult(
        annotations=[],
        messages=[],
    )
    assert result.annotations == []
    assert result.messages == []
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cached_input_tokens == 0
    assert result.model is None
    assert result.total_cost_usd is None


def test_document_extraction_result_stores_reported_cost() -> None:
    """total_cost_usd is the caller-supplied reported cost, not a re-estimate."""
    result = DocumentExtractionResult(
        annotations=[_sample_annotation()],
        messages=[{"role": "user", "content": "x"}],
        model="gpt-5.6-luna",
        input_tokens=10,
        output_tokens=5,
        cached_input_tokens=4,
        total_cost_usd=0.0015,
    )
    assert result.total_cost_usd == pytest.approx(0.0015)
    assert result.cached_input_tokens == 4


def test_document_extraction_result_cost_stays_none_without_reported_cost() -> None:
    """A model and token counts do not invent a cost when none was reported."""
    result = DocumentExtractionResult(
        annotations=[],
        messages=[],
        model="unknown-model",
        input_tokens=1,
        output_tokens=1,
    )
    assert result.total_cost_usd is None
    assert result.cached_input_tokens == 0


class TestDocumentExtractionResultCombine:
    """Tests for folding per-call results into one document result."""

    @staticmethod
    def _make_part(
        *,
        model: str | None = None,
        annotations: Sequence[GoldStandardAnnotation] = (),
        messages: Sequence[dict] = (),
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        llm_call_seconds: float = 0.0,
        total_cost_usd: float | None = None,
    ) -> DocumentExtractionResult:
        """Build one per-call result for combine tests."""
        return DocumentExtractionResult(
            annotations=list(annotations),
            messages=list(messages),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            llm_call_seconds=llm_call_seconds,
            total_cost_usd=total_cost_usd,
        )

    def test_empty_parts_yields_empty_result(self) -> None:
        """Combining no parts yields the empty result, model and cost included."""
        combined = DocumentExtractionResult.combine([])
        assert combined.annotations == []
        assert combined.messages == []
        assert combined.input_tokens == 0
        assert combined.output_tokens == 0
        assert combined.llm_call_seconds == 0.0
        assert combined.model is None
        assert combined.total_cost_usd is None

    def test_single_part_round_trips(self) -> None:
        """Combining one part reproduces every field of that part."""
        part = self._make_part(
            model="model-a",
            annotations=[_sample_annotation(1)],
            messages=[{"role": "user", "content": "x"}],
            input_tokens=10,
            output_tokens=5,
            llm_call_seconds=2.5,
        )
        combined = DocumentExtractionResult.combine([part])
        assert combined.annotations == part.annotations
        assert combined.messages == part.messages
        assert combined.input_tokens == part.input_tokens
        assert combined.output_tokens == part.output_tokens
        assert combined.llm_call_seconds == part.llm_call_seconds
        assert combined.model == part.model
        assert combined.total_cost_usd == part.total_cost_usd

    def test_annotations_concatenate_in_part_order(self) -> None:
        """Annotations from all parts appear, in part order."""
        first = self._make_part(annotations=[_sample_annotation(1)])
        second = self._make_part(annotations=[_sample_annotation(2)])
        combined = DocumentExtractionResult.combine([first, second])
        assert combined.annotations == first.annotations + second.annotations

    def test_messages_concatenate_in_part_order(self) -> None:
        """Messages from all parts appear, in part order."""
        first = self._make_part(messages=[{"role": "user", "content": "a"}])
        second = self._make_part(messages=[{"role": "user", "content": "b"}])
        combined = DocumentExtractionResult.combine([first, second])
        assert combined.messages == first.messages + second.messages

    def test_tokens_and_seconds_sum(self) -> None:
        """Token counts, cached tokens, and call durations add across parts."""
        first = self._make_part(
            input_tokens=10,
            output_tokens=3,
            cached_input_tokens=4,
            llm_call_seconds=1.0,
        )
        second = self._make_part(
            input_tokens=7,
            output_tokens=2,
            cached_input_tokens=2,
            llm_call_seconds=0.5,
        )
        combined = DocumentExtractionResult.combine([first, second])
        assert combined.input_tokens == 17
        assert combined.output_tokens == 5
        assert combined.cached_input_tokens == 6
        assert combined.llm_call_seconds == pytest.approx(1.5)

    def test_same_model_dedupes(self) -> None:
        """Parts sharing one model combine to that model alone."""
        combined = DocumentExtractionResult.combine(
            [self._make_part(model="model-a"), self._make_part(model="model-a")]
        )
        assert combined.model == "model-a"

    def test_distinct_models_join_in_call_order(self) -> None:
        """Different models combine in first-seen order, joined by ' + '."""
        combined = DocumentExtractionResult.combine(
            [
                self._make_part(model="model-b"),
                self._make_part(model="model-a"),
                self._make_part(model="model-b"),
            ]
        )
        assert combined.model == "model-b + model-a"

    def test_all_none_models_give_none_model(self) -> None:
        """Parts without a model combine to no model."""
        combined = DocumentExtractionResult.combine(
            [self._make_part(model=None), self._make_part(model=None)]
        )
        assert combined.model is None

    def test_reported_costs_sum_and_unpriced_parts_are_skipped(self) -> None:
        """Reported costs add across parts; parts without a cost contribute nothing."""
        combined = DocumentExtractionResult.combine(
            [
                self._make_part(model="model-a", total_cost_usd=0.002),
                self._make_part(model="model-a", total_cost_usd=None),
                self._make_part(model="model-a", total_cost_usd=0.001),
            ]
        )
        assert combined.total_cost_usd == pytest.approx(0.003)

    def test_cost_none_when_all_parts_unknown(self) -> None:
        """Parts without a cost combine to no cost."""
        combined = DocumentExtractionResult.combine(
            [self._make_part(model=None), self._make_part(model=None)]
        )
        assert combined.total_cost_usd is None


def test_extraction_run_metadata_defaults() -> None:
    """ExtractionRunMetadata has sensible defaults."""
    meta = ExtractionRunMetadata()
    assert meta.model is None
    assert meta.total_input_tokens == 0
    assert meta.total_output_tokens == 0
    assert meta.total_cached_input_tokens == 0
    assert meta.total_cost_usd is None
    assert meta.per_document == {}
    assert "cache" in meta.notes.total_cached_input_tokens.lower()
    assert "cached_input_tokens" in meta.notes.per_document


def test_extraction_run_metadata_explicit_fields() -> None:
    """ExtractionRunMetadata stores aggregate batch fields."""
    meta = ExtractionRunMetadata(
        model="gpt-5.6-luna",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cached_input_tokens=40,
        total_cost_usd=0.05,
        per_document={
            "1": PerDocumentExtractionStats(
                input_tokens=100,
                output_tokens=50,
                cached_input_tokens=40,
            ),
        },
    )
    assert meta.model == "gpt-5.6-luna"
    assert meta.total_input_tokens == 100
    assert meta.total_output_tokens == 50
    assert meta.total_cached_input_tokens == 40
    assert meta.total_cost_usd == pytest.approx(0.05)
    assert meta.per_document["1"].input_tokens == 100
    assert meta.per_document["1"].cached_input_tokens == 40


def test_extraction_run_output_round_trip() -> None:
    """ExtractionRunOutput bundles documents and metadata."""
    doc = _sample_gold_annotated_document()
    meta = ExtractionRunMetadata(model="m", total_input_tokens=10)
    out = ExtractionRunOutput(annotated_documents=[doc], metadata=meta)
    assert len(out.annotated_documents) == 1
    assert out.annotated_documents[0].document.name == "doc.pdf"
    assert out.metadata.model == "m"
    assert out.metadata.total_input_tokens == 10


def test_pipeline_stage_notes_cover_all_stages() -> None:
    """Every pipeline stage enum member has a human-readable note."""
    notes = pipeline_stage_notes()
    for stage in ExtractionPipelineStage:
        assert stage.value in notes
        assert notes[stage.value] == PIPELINE_STAGE_NOTES[stage]
