"""Tests for LLM extraction output data models."""

from unittest.mock import MagicMock, patch

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
    DocumentExtractionResult,
    ExtractionRunMetadata,
    ExtractionRunOutput,
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
        output_data=True,
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
    assert result.model is None
    assert result.total_cost_usd is None


@patch("deet.data_models.extraction.estimate_cost_usd")
def test_document_extraction_result_computes_total_cost_usd(
    mock_estimate: MagicMock,
) -> None:
    """total_cost_usd merges prompt and completion parts from estimate_cost_usd."""
    mock_estimate.return_value = (0.001, 0.002)
    result = DocumentExtractionResult(
        annotations=[_sample_annotation()],
        messages=[{"role": "user", "content": "x"}],
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=5,
    )
    mock_estimate.assert_called_once_with(
        "gpt-4o-mini",
        prompt_tokens=10,
        completion_tokens=5,
    )
    assert result.total_cost_usd == pytest.approx(0.003)


@patch("deet.data_models.extraction.estimate_cost_usd")
def test_document_extraction_result_cost_none_when_estimate_unknown(
    mock_estimate: MagicMock,
) -> None:
    """When litellm cannot price either side, total_cost_usd stays None."""
    mock_estimate.return_value = (None, None)
    result = DocumentExtractionResult(
        annotations=[],
        messages=[],
        model="unknown-model",
        input_tokens=1,
        output_tokens=1,
    )
    assert result.total_cost_usd is None


@patch("deet.data_models.extraction.estimate_cost_usd")
def test_document_extraction_result_cost_partial_prompt_only(
    mock_estimate: MagicMock,
) -> None:
    """Only prompt cost present is used as total."""
    mock_estimate.return_value = (0.01, None)
    result = DocumentExtractionResult(
        annotations=[],
        messages=[],
        model="m",
        input_tokens=1,
        output_tokens=0,
    )
    assert result.total_cost_usd == pytest.approx(0.01)


def test_extraction_run_metadata_defaults() -> None:
    """ExtractionRunMetadata has sensible defaults."""
    meta = ExtractionRunMetadata()
    assert meta.model is None
    assert meta.total_input_tokens == 0
    assert meta.total_output_tokens == 0
    assert meta.total_cost_usd is None
    assert meta.per_document_tokens == {}


def test_extraction_run_metadata_explicit_fields() -> None:
    """ExtractionRunMetadata stores aggregate batch fields."""
    meta = ExtractionRunMetadata(
        model="gpt-4o-mini",
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=0.05,
        per_document_tokens={"1": {"input_tokens": 100, "output_tokens": 50}},
    )
    assert meta.model == "gpt-4o-mini"
    assert meta.total_input_tokens == 100
    assert meta.total_output_tokens == 50
    assert meta.total_cost_usd == pytest.approx(0.05)
    assert meta.per_document_tokens["1"]["input_tokens"] == 100


def test_extraction_run_output_round_trip() -> None:
    """ExtractionRunOutput bundles documents and metadata."""
    doc = _sample_gold_annotated_document()
    meta = ExtractionRunMetadata(model="m", total_input_tokens=10)
    out = ExtractionRunOutput(annotated_documents=[doc], metadata=meta)
    assert len(out.annotated_documents) == 1
    assert out.annotated_documents[0].document.name == "doc.pdf"
    assert out.metadata.model == "m"
    assert out.metadata.total_input_tokens == 10
