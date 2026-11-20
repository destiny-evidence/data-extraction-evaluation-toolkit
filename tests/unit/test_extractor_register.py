import sys
from unittest.mock import MagicMock, patch

import pytest

from deet.extractors.base_extractor import (
    BaseDataExtractor,
    DataExtractionConfig,
    ExtractionMethod,
)
from deet.extractors.extractor_registry import extractor_mapping, get_data_extractor
from deet.settings import DataExtractionSettings


@pytest.mark.parametrize("extraction_method", list(ExtractionMethod))
def test_extraction_methods_return_extractor(extraction_method):
    """Test that each member of ExtractionMethod returns a valid extractor."""
    config = DataExtractionConfig(method=extraction_method)
    settings = DataExtractionSettings(
        azure_api_key="key", azure_api_base="https://example.invalid"
    )
    with patch.dict(sys.modules, {"fastembed": MagicMock()}):
        extractor = get_data_extractor(config=config, settings=settings)
    assert isinstance(extractor, BaseDataExtractor)
    assert extractor.settings is settings


def test_every_extraction_method_is_registered():
    """The registry covers every ExtractionMethod member (guards new additions)."""
    assert set(extractor_mapping) == set(ExtractionMethod)
