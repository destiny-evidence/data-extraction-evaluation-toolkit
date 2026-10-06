"""Top down LLM Extractor that starts at root and descends through leaves."""

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

        context = self._prepare_context(payload=payload, context_type=context_type)

        all_annotations = []
        all_messages = []
        total_input_tokens = 0
        total_output_tokens = 0

        for scheme in self.mapped_schemes:
            logger.info(f"extracting concepts from scheme: {scheme.title}")
            leaves = scheme.roots
            level = 1
            while leaves:
                logger.info(
                    f"Extracting attributes at level {level} of concept hierarchy."
                )
                level_attributes = [concept.attribute for concept in leaves]
                prompt = self._generate_user_message_json(
                    payload=context, attributes=level_attributes
                )

                response_model: type[BaseLLMResponse]

                if self.config.dynamic_json_schema:
                    response_model = build_llm_response_model(level_attributes)
                else:
                    response_model = LLMResponseSchema
                llm_response, messages, output_tokens, input_tokens = self._call_llm(
                    prompt=prompt, response_model=response_model
                )
                all_messages.extend(messages)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                annotations = self._parse_llm_response(
                    response_content=llm_response,
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
                leaves = [
                    child
                    for concept in leaves
                    if concept.attribute.attribute_id in present_attribute_ids
                    for child in scheme.narrower(concept.identifier)
                ]
                logger.info(
                    (
                        "Based on the output above,"
                        " the following child attributes will be prompted for:\n{}"
                    ),
                    pretty_repr(leaves),
                )
                level += 1

        return DocumentExtractionResult(
            annotations=all_annotations,
            messages=all_messages,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=self.model,
        )
