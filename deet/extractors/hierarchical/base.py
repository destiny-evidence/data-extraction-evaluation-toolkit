"""Extensible LLM extractors for hierarchical attributes."""

from collections.abc import Sequence
from pathlib import Path

from deet.data_models.base import Attribute
from deet.data_models.documents import ContextType, Document
from deet.data_models.extraction import DocumentParsingStats, ExtractionRunOutput
from deet.data_models.taxonomy import ConceptScheme
from deet.extractors.llm_data_extractor import LLMDataExtractor


class VocabularyLLMExtractor(LLMDataExtractor):
    """LLM-based data extractor for hierarchical attribute data."""

    def load_schemes(self) -> list[ConceptScheme]:
        """Read the vocabulary attached to the project."""
        from deet.data_models.taxonomy import load_schemes_from_ttl

        if self.config.vocabulary_path:
            return load_schemes_from_ttl(self.config.vocabulary_path)
        return []

    def extract_from_documents(  # noqa: PLR0913
        self,
        attributes: list[Attribute],
        documents: Sequence[Document],
        filter_attribute_ids: list[int] | None = None,
        output_file: Path | None = None,
        context_type: ContextType | None = None,
        prompt_outfile: Path | None = None,
        document_parsing: dict[str, DocumentParsingStats] | None = None,
        *,
        show_progress: bool = False,
    ) -> ExtractionRunOutput:
        """Map attributes to vocab, and then call inherited extract_from_documents."""
        self.mapped_schemes = [
            scheme.map_concepts(
                mapping_file=self.config.vocabulary_mapping_path, attributes=attributes
            )
            for scheme in self.load_schemes()
        ]
        return super().extract_from_documents(
            attributes,
            documents,
            filter_attribute_ids,
            output_file,
            context_type,
            prompt_outfile,
            show_progress=show_progress,
        )
