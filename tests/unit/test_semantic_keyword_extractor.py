"""Tests for the semantic keyword data extractor module."""

from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from deet.data_models.base import AnnotationType, Attribute, AttributeType
from deet.data_models.documents import ContextType
from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.extractors.keyword.semantic_keyword_extractor import (
    SemanticKeywordDataExtractor,
)


@pytest.fixture
def config() -> DataExtractionConfig:
    """Config selecting the semantic keyword extraction method."""
    return DataExtractionConfig(
        method=ExtractionMethod.SEMANTIC,
        default_context_type=ContextType.FULL_DOCUMENT,
    )


def _attribute(attribute_id: int, prompt: str | None) -> Attribute:
    return Attribute(
        attribute_id=attribute_id,
        attribute_label=f"Attribute {attribute_id}",
        output_data_type=AttributeType.BOOL,
        prompt=prompt,
    )


def _make_extractor(config, chunk_emb, keyword_emb) -> SemanticKeywordDataExtractor:
    """Build an extractor whose model returns the supplied embeddings in order."""
    model = MagicMock()
    model.encode.side_effect = [np.array(chunk_emb), np.array(keyword_emb)]
    with patch(
        "deet.extractors.keyword.semantic_keyword_extractor.SentenceTransformer",
        return_value=model,
    ):
        return SemanticKeywordDataExtractor(config=config)


def test_split_into_sentences(config):
    """Sentences are split on terminal punctuation followed by a space."""
    extractor = _make_extractor(config, [[1.0]], [[1.0]])
    sentences = extractor._split_into_sentences("First one. Second two! Third?")
    assert sentences == ["First one.", "Second two!", "Third?"]


def test_similarity_above_threshold_produces_annotation(config):
    """A chunk similar enough to a prompt yields an annotation with reasoning."""
    extractor = _make_extractor(
        config,
        chunk_emb=[[1.0, 0.0], [0.0, 1.0]],
        keyword_emb=[[1.0, 0.0]],
    )
    result = extractor.extract_from_document(
        attributes=[_attribute(1, "climate")],
        payload="Sentence one. Sentence two.",
    )
    assert len(result.annotations) == 1
    annotation = result.annotations[0]
    assert annotation.attribute.attribute_id == 1
    assert annotation.annotation_type == AnnotationType.KEYWORD
    assert annotation.reasoning is not None
    assert "Sentence one." in annotation.reasoning


def test_similarity_below_threshold_produces_no_annotation(config):
    """A chunk dissimilar to every prompt yields no annotation."""
    extractor = _make_extractor(
        config,
        chunk_emb=[[0.0, 1.0]],
        keyword_emb=[[1.0, 0.0]],
    )
    result = extractor.extract_from_document(
        attributes=[_attribute(1, "climate")],
        payload="Unrelated sentence.",
    )
    assert result.annotations == []


def test_empty_document_yields_no_annotations(config):
    """A document with no sentences returns early without encoding."""
    extractor = _make_extractor(config, [[1.0]], [[1.0]])
    result = extractor.extract_from_document(
        attributes=[_attribute(1, "climate")],
        payload="",
    )
    assert result.annotations == []
    cast("MagicMock", extractor.model).encode.assert_not_called()


def test_attribute_without_prompt_raises(config):
    """An attribute with no usable prompt is a misconfiguration, not skipped."""
    extractor = _make_extractor(config, [[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="non-empty prompt"):
        extractor.extract_from_document(
            attributes=[_attribute(1, None)],
            payload="Some sentence here.",
        )
