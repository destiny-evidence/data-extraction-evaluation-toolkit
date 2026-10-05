"""Registry of extraction methods and their extractors."""

from deet.extractors.base_extractor import (
    BaseDataExtractor,
    DataExtractionConfig,
    ExtractionMethod,
)
from deet.extractors.keyword.raw_keyword_extractor import RawKeywordDataExtractor
from deet.extractors.keyword.semantic_keyword_extractor import (
    SemanticKeywordDataExtractor,
)
from deet.extractors.llm_data_extractor import LLMDataExtractor

extractor_mapping: dict[ExtractionMethod, type[BaseDataExtractor]] = {
    ExtractionMethod.LLM: LLMDataExtractor,
    ExtractionMethod.KEYWORD: RawKeywordDataExtractor,
    ExtractionMethod.SEMANTIC: SemanticKeywordDataExtractor,
}


def get_data_extractor(config: DataExtractionConfig) -> BaseDataExtractor:
    """Instantiate the extractor registered for the given method."""
    return extractor_mapping[config.method](config=config)
