"""Tests for LLM extraction output data models."""

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


def _sample_attribute() -> Attribute:
    """Build a minimal valid Attribute for tests."""
    return Attribute(
        prompt="Q?",
        output_data_type=AttributeType.BOOL,
        attribute_id=1,
        attribute_label="Attr 1",
    )


def _sample_annotation() -> GoldStandardAnnotation:
    """Build a minimal GoldStandardAnnotation."""
    return GoldStandardAnnotation(
        attribute=_sample_attribute(),
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
