"""Shared classes for keyword-type extractors."""

from deet.data_models.base import Attribute, AttributeType
from deet.extractors.base_extractor import BaseDataExtractor


class BaseKeywordDataExtractor(BaseDataExtractor):
    """
    Intermediate extractor class for keyword extractors.

    Provides shared phrase-parsing helpers.
    Subclasses must implement `extract_from_document`
    (as do all concrete implementations of `BaseDataExtractor`)
    """

    PROMPT_SEPARATOR = ";"
    SUPPORTED_ATTRIBUTE_TYPES = frozenset({AttributeType.BOOL})

    def _get_prompt_phrases(self, attribute: Attribute) -> list[str]:
        """
        Split an attribute's prompt into phrases by PROMPT_SEPARATOR.

        Drop any empty phrases.
        """
        if not attribute.prompt:
            return []
        phrases = attribute.prompt.split(self.PROMPT_SEPARATOR)
        return [stripped for p in phrases if (stripped := p.strip())]
