"""Shared classes for keyword-type extractors."""

from deet.data_models.base import Attribute
from deet.extractors.base_extractor import BaseDataExtractor


class BaseKeywordDataExtractor(BaseDataExtractor):
    """ABC for keyword data extractors, handling tasks common to them."""

    PROMPT_SEPARATOR = ";"

    def _get_prompt_phrases(self, attribute: Attribute) -> list[str]:
        """Split an attribute's prompt into phrases by PROMPT_SEPARATOR."""
        if not attribute.prompt:
            return []
        phrases = attribute.prompt.split(self.PROMPT_SEPARATOR)
        return [stripped for p in phrases if (stripped := p.strip())]
