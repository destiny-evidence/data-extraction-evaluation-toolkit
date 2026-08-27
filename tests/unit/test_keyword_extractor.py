"""Tests for the keyword data extractor module."""

import pytest

from deet.data_models.base import AnnotationType, Attribute, AttributeType
from deet.data_models.documents import ContextType
from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.extractors.keyword_extractor import KeywordDataExtractor


@pytest.fixture
def config() -> DataExtractionConfig:
    """Config selecting the keyword extraction method."""
    return DataExtractionConfig(
        method=ExtractionMethod.KEYWORD,
        default_context_type=ContextType.FULL_DOCUMENT,
    )


@pytest.fixture
def extractor(config) -> KeywordDataExtractor:
    """Keyword extractor under test."""
    return KeywordDataExtractor(config=config)


def _attribute(attribute_id: int, prompt: str | None) -> Attribute:
    return Attribute(
        attribute_id=attribute_id,
        attribute_label=f"Attribute {attribute_id}",
        output_data_type=AttributeType.BOOL,
        prompt=prompt,
    )


def test_matching_term_produces_annotation(extractor):
    """A prompt term present in the payload yields one keyword annotation."""
    attributes = [_attribute(1, "climate")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="the text discusses climate policy",
    )
    assert len(result.annotations) == 1
    annotation = result.annotations[0]
    assert annotation.attribute.attribute_id == 1
    assert annotation.raw_data is True
    assert annotation.annotation_type == AnnotationType.KEYWORD
    assert result.messages == []


def test_absent_term_produces_no_annotation(extractor):
    """A prompt term absent from the payload yields no annotation."""
    attributes = [_attribute(1, "climate")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="the text discusses fiscal policy",
    )
    assert result.annotations == []


def test_matches_at_most_once_per_attribute(extractor):
    """Multiple matching terms for one attribute produce a single annotation."""
    attributes = [_attribute(1, "climate change")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="climate change appears here",
    )
    assert len(result.annotations) == 1
