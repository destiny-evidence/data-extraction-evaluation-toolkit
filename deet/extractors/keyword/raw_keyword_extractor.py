"""Plain string-matching keyword extractor."""

from pathlib import Path

from deet.data_models.base import AnnotationType, Attribute, GoldStandardAnnotation
from deet.data_models.documents import (
    ContextType,
)
from deet.data_models.extraction import (
    DocumentExtractionResult,
)
from deet.extractors.keyword.base_keyword_extractor import BaseKeywordDataExtractor


class RawKeywordDataExtractor(BaseKeywordDataExtractor):
    """
    String-matching implementation of a keyword extractor.

    Extracts the presence of attributes where phrases in attribute prompts
    match the document content.
    """

    SNIPPET_WINDOW = 80

    def _snippet_around(self, context: str, idx: int, phrase_len: int) -> str:
        """Return a window of the document around the first match of phrase."""
        start = max(0, idx - self.SNIPPET_WINDOW)
        end = min(len(context), idx + phrase_len + self.SNIPPET_WINDOW)
        snippet = context[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(context) else ""
        return f"{prefix}{snippet}{suffix}"

    def extract_from_document(
        self,
        attributes: list[Attribute],
        filter_attribute_ids: list[int] | None = None,
        *,
        payload: str | None = None,
        md_path: Path | None = None,
        context_type: ContextType | None = None,
    ) -> DocumentExtractionResult:
        """
        Extract data from a single document.

        Return Annotations with output_data=`True` for all attributes where
        any phrase in the attribute's prompt (phrases are separated by ';')
        is contained in the document.
        """
        payload = self._resolve_payload(payload=payload, md_path=md_path)

        selected_attributes = self._select_attributes(attributes, filter_attribute_ids)

        context = self._prepare_context(payload=payload, context_type=context_type)
        context_lower = context.lower()
        annotations: list[GoldStandardAnnotation] = []
        for attribute in selected_attributes:
            prompt = attribute.prompt
            if prompt is None:
                continue
            for phrase in self._get_prompt_phrases(attribute):
                idx = context_lower.find(phrase.lower())
                if idx != -1:
                    annotations.extend(
                        [
                            GoldStandardAnnotation(
                                attribute=attribute,
                                raw_data=True,
                                annotation_type=AnnotationType.KEYWORD,
                                additional_text=self._snippet_around(
                                    context, idx, len(phrase)
                                ),
                                reasoning=f"Matched phrase '{phrase}'",
                            )
                        ]
                    )
                    break

        return DocumentExtractionResult(annotations=annotations, messages=[])
