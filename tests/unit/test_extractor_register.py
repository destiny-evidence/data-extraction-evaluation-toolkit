from unittest.mock import patch

import pytest

from deet.extractors.base_extractor import (
    BaseDataExtractor,
    DataExtractionConfig,
    ExtractionMethod,
)
from deet.extractors.extractor_registry import get_data_extractor


@pytest.mark.parametrize("extraction_method", list(ExtractionMethod))
def test_extraction_methods_return_extractor(extraction_method):
    """Test that each member of ExtractionMethod returns a valid extractor."""
    config = DataExtractionConfig(method=extraction_method)
    with patch("deet.extractors.semantic_keyword_extractor.SentenceTransformer"):
        extractor = get_data_extractor(config=config)
    assert isinstance(extractor, BaseDataExtractor)
