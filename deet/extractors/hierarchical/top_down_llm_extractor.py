"""
Top down LLM Extraction over a vocabulary hierarchy.

Asks about each scheme's root concepts first, then about the children
of concepts judged to apply, and then their children, all the way to the leaves
(concepts with no children).

Sub-trees of non-applying concepts are never queried.

See docs/concepts/vocabulary.md for more information.
"""

from pathlib import Path

from loguru import logger
from rich.pretty import pretty_repr

from deet.data_models.base import (
    Attribute,
    BaseLLMResponse,
    LLMResponseSchema,
    build_llm_response_model,
)
from deet.data_models.documents import (
    ContextType,
)
from deet.data_models.extraction import (
    DocumentExtractionResult,
)
from deet.extractors.hierarchical.base import VocabularyLLMExtractor
from deet.settings import (
    get_settings,
)

settings = get_settings()


class TopDownLLMExtractor(VocabularyLLMExtractor):
    """
    LLM-based data extractor for hierarchical attribute data.

    Start from the top level of a vocabulary, query for membership of the next level.

    Continue descending tree, excluding branches where
    attribute is not present.
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
        """
        Extract data from a single document.

        Call with either payload (document text) or md_path (path to markdown file).
        If md_path is provided, the file is read and used as the payload.
        Prompt payloads are not written here; the batch entry point
        extract_from_documents writes them to prompt_outfile when provided.

        Args:
            attributes: List of attributes to extract.
            payload: Document text to extract from. Required if md_path not set.
            md_path: Path to a markdown file to read as payload.
                Required if payload not set.
            context_type: Override config context type; if None, use config default.

        Returns:
            DocumentExtractionResult with annotations, messages, token counts,
            cost, model name, and timestamp.

        Raises:
            ValueError: If no attributes are selected for extraction after filtering.
            ValueError: If neither payload nor md_path provided, or both provided.

        """
        payload = self._resolve_payload(payload=payload, md_path=md_path)

        self._select_attributes(attributes, filter_attribute_ids)

        context = self._prepare_context(payload=payload, context_type=context_type)

        all_annotations = []
        all_messages = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_input_tokens = 0
        total_cost_usd: float | None = 0.0

        for scheme in self.mapped_schemes:
            logger.info(f"extracting concepts from scheme: {scheme.title}")
            level_concepts = scheme.roots
            level = 1
            while level_concepts:
                logger.info(
                    f"Extracting attributes at level {level} of concept hierarchy."
                )
                level_attributes = [concept.attribute for concept in level_concepts]
                prompt = self._generate_user_message_json(
                    payload=context, attributes=level_attributes
                )

                response_model: type[BaseLLMResponse]

                if self.config.dynamic_json_schema:
                    response_model = build_llm_response_model(level_attributes)
                else:
                    response_model = LLMResponseSchema
                llm_call = self._call_llm(prompt=prompt, response_model=response_model)
                all_messages.extend(llm_call.messages)
                total_input_tokens += llm_call.input_tokens
                total_output_tokens += llm_call.output_tokens
                total_cached_input_tokens += llm_call.cached_input_tokens
                if llm_call.total_cost_usd is None:
                    total_cost_usd = None
                elif total_cost_usd is not None:
                    total_cost_usd += llm_call.total_cost_usd
                annotations = self._parse_llm_response(
                    response_content=llm_call.response_content,
                    attributes=level_attributes,
                    response_model=response_model,
                )
                logger.debug(
                    "Model returned the following annotations:\n{}",
                    pretty_repr(annotations),
                )
                all_annotations.extend(annotations)
                present_attribute_ids = {
                    ann.attribute.attribute_id for ann in annotations if ann.output_data
                }
                level_concepts = [
                    child
                    for concept in level_concepts
                    if concept.attribute.attribute_id in present_attribute_ids
                    for child in scheme.narrower(concept.identifier)
                ]
                logger.info(
                    (
                        "Based on the output above,"
                        " the following child attributes will be prompted for:\n{}"
                    ),
                    pretty_repr(level_concepts),
                )
                level += 1

        return DocumentExtractionResult(
            annotations=all_annotations,
            messages=all_messages,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=self.model,
        )
