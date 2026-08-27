"""Test that all extractors implement desired behaviour."""

from unittest.mock import patch

import pytest

from deet.data_models.base import Attribute, AttributeType
from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.extractors.extractor_registry import get_data_extractor


def _attr(attribute_id: int) -> Attribute:
    return Attribute(
        attribute_id=attribute_id,
        attribute_label=f"Attribute {attribute_id}",
        output_data_type=AttributeType.BOOL,
    )


@pytest.fixture(params=list(ExtractionMethod), ids=lambda m: m.value)
def extractor(request):
    """Each registered extractor, constructed via the registry."""
    config = DataExtractionConfig(method=request.param)
    with patch(
        "deet.extractors.keyword.semantic_keyword_extractor.SentenceTransformer"
    ):
        return get_data_extractor(config=config)


def test_neither_payload_nor_md_path_raises(extractor):
    with pytest.raises(ValueError, match="Exactly one of payload or md_path"):
        extractor.extract_from_document(attributes=[])


def test_both_payload_and_md_path_raises(extractor, tmp_path):
    with pytest.raises(ValueError, match="Exactly one of payload or md_path"):
        extractor.extract_from_document(
            attributes=[], payload="x", md_path=tmp_path / "doc.md"
        )


def test_missing_md_path_raises(extractor, tmp_path):
    with pytest.raises(FileNotFoundError, match="Markdown file not found"):
        extractor.extract_from_document(attributes=[], md_path=tmp_path / "missing.md")


def test_empty_attributes_raises(extractor):
    with pytest.raises(ValueError, match="No attributes selected"):
        extractor.extract_from_document(attributes=[], payload="anything")


def test_select_attributes_returns_all_when_no_filter(extractor):
    attrs = [_attr(1), _attr(2)]
    assert extractor._select_attributes(attrs, None) == attrs


def test_select_attributes_filters_to_matching_ids(extractor):
    attrs = [_attr(1), _attr(2), _attr(3)]
    result = extractor._select_attributes(attrs, [1, 3])
    assert [a.attribute_id for a in result] == [1, 3]
