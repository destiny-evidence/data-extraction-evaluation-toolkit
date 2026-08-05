"""Registry of extraction methods and their extractors."""

from deet.extractors.base_extractor import (
    BaseDataExtractor,
    DataExtractionConfig,
    ExtractionMethod,
)
from deet.extractors.hierarchical.top_down_llm_extractor import TopDownLLMExtractor
from deet.extractors.keyword_extractor import KeywordDataExtractor
from deet.extractors.llm_data_extractor import LLMDataExtractor
from deet.extractors.semantic_keyword_extractor import SemanticKeywordDataExtractor

extractor_mapping: dict[ExtractionMethod, type[BaseDataExtractor]] = {
    ExtractionMethod.LLM: LLMDataExtractor,
    ExtractionMethod.KEYWORD: KeywordDataExtractor,
    ExtractionMethod.SEMANTIC: SemanticKeywordDataExtractor,
    ExtractionMethod.HIERARCHICAL_TOP_DOWN: TopDownLLMExtractor,
}


def get_data_extractor(config: DataExtractionConfig) -> BaseDataExtractor:
    """Instantiate the extractor registered for the given method."""
    return extractor_mapping[config.method](config=config)
