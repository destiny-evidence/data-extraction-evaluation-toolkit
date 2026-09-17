"""Tests for the keyword data extractor module."""

import pytest

from deet.data_models.base import AnnotationType
from deet.data_models.documents import ContextType
from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.extractors.keyword.raw_keyword_extractor import RawKeywordDataExtractor


@pytest.fixture
def config() -> DataExtractionConfig:
    """Config selecting the keyword extraction method."""
    return DataExtractionConfig(
        method=ExtractionMethod.KEYWORD,
        default_context_type=ContextType.FULL_DOCUMENT,
    )


@pytest.fixture
def extractor(config) -> RawKeywordDataExtractor:
    """Keyword extractor under test."""
    return RawKeywordDataExtractor(config=config)


def test_matching_term_produces_annotation(extractor, make_attr):
    """A prompt term present in the payload yields one keyword annotation."""
    attributes = [make_attr(1, "climate")]
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


def test_absent_term_produces_no_annotation(extractor, make_attr):
    """A prompt term absent from the payload yields no annotation."""
    attributes = [make_attr(1, "climate")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="the text discusses fiscal policy",
    )
    assert result.annotations == []


def test_matches_at_most_once_per_attribute(extractor, make_attr):
    """Multiple matching phrases for one attribute produce a single annotation."""
    attributes = [make_attr(1, "climate; change")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="climate change appears here",
    )
    assert len(result.annotations) == 1


def test_matches_when_any_phrase_present(extractor, make_attr):
    """An attribute matches if any of its separated phrases appears."""
    attributes = [make_attr(1, "renewable energy; climate adaptation")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="the study covers climate adaptation in cities",
    )
    assert len(result.annotations) == 1


def test_phrase_must_match_as_a_whole(extractor, make_attr):
    """A multi-word phrase matches as a unit, not by its individual words."""
    attributes = [make_attr(1, "climate change")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="this is about climate policy and social change",
    )
    assert result.annotations == []


def test_separator_only_prompt_does_not_match(extractor, make_attr):
    """A prompt of only separators yields no phrases and no false positives."""
    attributes = [make_attr(1, ";;")]
    result = extractor.extract_from_document(
        attributes=attributes,
        payload="any text at all",
    )
    assert result.annotations == []


def test_get_prompt_phrases_splits_strips_and_drops_empties(extractor, make_attr):
    """Prompts split on the separator, are stripped, and empties are dropped."""
    assert extractor._get_prompt_phrases(make_attr(1, "a; b ; c")) == ["a", "b", "c"]
    assert extractor._get_prompt_phrases(make_attr(1, "climate change")) == [
        "climate change"
    ]
    assert extractor._get_prompt_phrases(make_attr(1, " ; ; ")) == []
    assert extractor._get_prompt_phrases(make_attr(1, None)) == []
