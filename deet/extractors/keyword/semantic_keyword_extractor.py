"""Semantic keyword extractor using sentence-transformer embeddings."""

import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from deet.data_models.base import AnnotationType, Attribute, GoldStandardAnnotation
from deet.data_models.documents import ContextType
from deet.data_models.extraction import DocumentExtractionResult
from deet.extractors.base_extractor import DataExtractionConfig
from deet.extractors.keyword.base_keyword_extractor import BaseKeywordDataExtractor


class SemanticKeywordDataExtractor(BaseKeywordDataExtractor):
    """
    Keyword extractor matching attribute phrases to document sentences.

    Uses sentence-transformer embeddings and cosine similarity rather than
    exact string matching, so semantically related wording still matches.
    """

    def __init__(self, config: DataExtractionConfig) -> None:
        """Initialise, set the similarity threshold, and load the model."""
        super().__init__(config)
        self.similarity_threshold: float = config.semantic_similarity_threshold
        self.model = SentenceTransformer(config.model)

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences on terminal punctuation followed by a space."""
        sentences = re.split(r"(?<=[.!?]) +", text)
        return [s.strip() for s in sentences if s.strip()]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a (len(texts), dim) array of vectors."""
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _phrases_per_attribute(self, attributes: list[Attribute]) -> list[list[str]]:
        """
        Return the keyword phrases for each attribute, in order.

        A keyword attribute with no usable prompt has nothing to match on, so an
        empty result is presumed to be an error.

        Raises:
            ValueError: If any attribute yields no phrases.

        """
        phrases_per_attribute = [self._get_prompt_phrases(attr) for attr in attributes]
        if any(not phrases for phrases in phrases_per_attribute):
            msg = "Every attribute must have a non-empty prompt for keyword extraction"
            raise ValueError(msg)
        return phrases_per_attribute

    def _group_similarities_by_attribute(
        self,
        similarity_matrix: np.ndarray,
        phrases_per_attribute: list[list[str]],
    ) -> list[np.ndarray]:
        """
        Split the phrase-by-chunk similarity matrix into one block per attribute.

        Every attribute's phrases are scored in a single matrix, with their rows
        stacked in order. Using each attribute's phrase count, we slice those
        rows back into per-attribute blocks, so each attribute is scored only
        against its own phrases.
        """
        group_sizes = [len(phrases) for phrases in phrases_per_attribute]
        split_points = np.cumsum(group_sizes)[:-1]
        return np.split(similarity_matrix, split_points, axis=0)

    def _best_sentence_match(
        self,
        similarities: np.ndarray,
        phrases: list[str],
        doc_chunks: list[str],
    ) -> tuple[float, str, str]:
        """Return the best (score, sentence) from one attribute's similarity rows."""
        phrase_idx, chunk_idx = np.unravel_index(
            similarities.argmax(), similarities.shape
        )
        return (
            float(similarities.max()),
            phrases[int(phrase_idx)],
            doc_chunks[chunk_idx],
        )

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

        Return Annotations with output_data=`True` for all attributes where the maximum
        similarity between a phrase ("separated by ';') in the prompt and the document
        if greater than `DataExtractionConfig.semantic_similarity_threshold`
        """
        payload = self._resolve_payload(payload=payload, md_path=md_path)
        selected_attributes = self._select_attributes(attributes, filter_attribute_ids)
        context = self._prepare_context(payload=payload, context_type=context_type)
        doc_chunks = self._split_into_sentences(context)
        if not doc_chunks:
            return DocumentExtractionResult(annotations=[], messages=[])

        phrases_per_attribute = self._phrases_per_attribute(selected_attributes)

        chunk_embeddings = self._encode(doc_chunks)
        phrase_embeddings = self._encode(
            [phrase for phrases in phrases_per_attribute for phrase in phrases]
        )
        similarity_matrix = cosine_similarity(phrase_embeddings, chunk_embeddings)
        similarities_per_attribute = self._group_similarities_by_attribute(
            similarity_matrix, phrases_per_attribute
        )

        annotations: list[GoldStandardAnnotation] = []
        for attribute, phrases, similarities in zip(
            selected_attributes,
            phrases_per_attribute,
            similarities_per_attribute,
            strict=True,
        ):
            score, phrase, sentence = self._best_sentence_match(
                similarities, phrases, doc_chunks
            )
            if score >= self.similarity_threshold:
                annotations.append(
                    GoldStandardAnnotation(
                        attribute=attribute,
                        raw_data=True,
                        annotation_type=AnnotationType.KEYWORD,
                        additional_text=sentence,
                        reasoning=f"Phrase '{phrase}' matched with score {score:.4f}",
                    )
                )

        return DocumentExtractionResult(annotations=annotations, messages=[])
