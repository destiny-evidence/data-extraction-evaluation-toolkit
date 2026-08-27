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

    def extract_from_document(
        self,
        attributes: list[Attribute],
        filter_attribute_ids: list[int] | None = None,
        *,
        payload: str | None = None,
        md_path: Path | None = None,
        context_type: ContextType | None = None,
    ) -> DocumentExtractionResult:
        """Extract data from a single document."""
        payload = self._resolve_payload(payload=payload, md_path=md_path)

        selected_attributes = self._select_attributes(attributes, filter_attribute_ids)

        context = self._prepare_context(payload=payload, context_type=context_type)
        annotations: list[GoldStandardAnnotation] = []
        for attribute in selected_attributes:
            prompt = attribute.prompt
            if prompt is None:
                continue
            for phrase in self._get_prompt_phrases(attribute):
                if phrase.lower().strip() in context.lower():
                    annotations.extend(
                        [
                            GoldStandardAnnotation(
                                attribute=attribute,
                                raw_data=True,
                                annotation_type=AnnotationType.KEYWORD,
                            )
                        ]
                    )
                    break

        return DocumentExtractionResult(annotations=annotations, messages=[])
