"""Generalisable data extraction module for keyword-based extraction."""

from pathlib import Path
from typing import cast

from loguru import logger

from deet.data_models.base import AnnotationType, Attribute, GoldStandardAnnotation
from deet.data_models.documents import (
    ContextType,
)
from deet.data_models.extraction import (
    DocumentExtractionResult,
)
from deet.extractors.base_extractor import BaseDataExtractor


class KeywordDataExtractor(BaseDataExtractor):
    """
    Generalisable module for LLM-based data extraction from documents.

    This module provides a flexible interface for extracting structured data
    from documents using LLMs, with support for different context types and
    customizable prompts.
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
        if (payload is None and md_path is None) or (
            payload is not None and md_path is not None
        ):
            msg = "Exactly one of payload or md_path must be provided"
            raise ValueError(msg)
        if md_path is not None:
            if not md_path.exists():
                msg = f"Markdown file not found: {md_path}"
                raise FileNotFoundError(msg)
            payload = md_path.read_text(encoding="utf-8")
        payload = cast("str", payload)

        selected_attributes = attributes
        # TODO: Implement attribute filtering as method of ABC extractor

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(payload=payload, context_type=context_type)
        annotations: list[GoldStandardAnnotation] = []
        for attribute in attributes:
            prompt = attribute.prompt
            if prompt is None:
                continue
            for term in prompt.split():
                if term.lower().strip() in context:
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
