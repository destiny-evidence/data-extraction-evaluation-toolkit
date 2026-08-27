"""Test that all extractors implement desired behaviour."""

from unittest.mock import patch

import pytest

from deet.extractors.base_extractor import DataExtractionConfig, ExtractionMethod
from deet.extractors.extractor_registry import get_data_extractor


@pytest.fixture(params=list(ExtractionMethod), ids=lambda m: m.value)
def extractor(request):
    """Each registered extractor, constructed via the registry."""
    config = DataExtractionConfig(method=request.param)
    with patch("deet.extractors.semantic_keyword_extractor.SentenceTransformer"):
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
