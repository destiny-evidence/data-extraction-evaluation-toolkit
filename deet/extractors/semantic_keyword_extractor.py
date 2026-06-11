"""Generalisable data extraction module for semantic ^keyword-based extraction."""

import re
from pathlib import Path
from typing import cast

from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from deet.data_models.base import AnnotationType, Attribute, GoldStandardAnnotation
from deet.data_models.documents import (
    ContextType,
)
from deet.data_models.extraction import (
    DocumentExtractionResult,
)
from deet.extractors.base_extractor import BaseDataExtractor, DataExtractionConfig


class SemanticKeywordDataExtractor(BaseDataExtractor):
    """
    Generalisable module for LLM-based data extraction from documents.

    This module provides a flexible interface for extracting structured data
    from documents using LLMs, with support for different context types and
    customizable prompts.
    """

    def __init__(
        self,
        config: DataExtractionConfig,
    ) -> None:
        """Initialise, set threshold, and load model."""
        self.config = config
        self.similarity_threshold: float = 0.65
        self.model = SentenceTransformer(config.model)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split document into setences."""
        sentences = re.split(r"(?<=[.!?]) +", text)
        return [s.strip() for s in sentences if s.strip()]

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

        annotations: list[GoldStandardAnnotation] = []

        selected_attributes = attributes
        # TODO: Implement attribute filtering as method of ABC extractor

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(payload=payload, context_type=context_type)
        doc_chunks = self._split_into_sentences(context)

        target_phrases = [attr.prompt for attr in attributes]

        chunk_embeddings = self.model.encode(
            doc_chunks, show_progress_bar=False, convert_to_numpy=True
        )
        keyword_embeddings = self.model.encode(
            target_phrases, show_progress_bar=False, convert_to_numpy=True
        )

        similarity_matrix = cosine_similarity(keyword_embeddings, chunk_embeddings)

        for attr_idx, attribute in enumerate(attributes):
            max_sim_score = similarity_matrix[attr_idx].max()
            if max_sim_score >= self.similarity_threshold:
                best_match_idx = similarity_matrix[attr_idx].argmax()
                sentence = doc_chunks[best_match_idx]
                annotations.append(
                    GoldStandardAnnotation(
                        attribute=attribute,
                        raw_data=True,
                        annotation_type=AnnotationType.KEYWORD,
                        reasoning=f"Matches {sentence}, score {max_sim_score:.4f}",
                    )
                )

        return DocumentExtractionResult(annotations=annotations, messages=[])
