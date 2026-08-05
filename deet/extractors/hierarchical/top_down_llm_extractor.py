"""Top down LLM Extractor that starts at root and descends through leaves."""

from pathlib import Path
from typing import cast

from loguru import logger

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
        if filter_attribute_ids and len(filter_attribute_ids) > 0:
            try:
                selected_attributes = self._filter_attributes(
                    selected_attributes, filter_ids=filter_attribute_ids
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid attribute IDs in config: "
                    f"{filter_attribute_ids}. "
                    "No attributes will be selected."
                )

        if not selected_attributes:
            msg = "No attributes selected for extraction"
            logger.warning(msg)
            raise ValueError(msg)

        context = self._prepare_context(payload=payload, context_type=context_type)

        all_annotations = []
        all_messages = []
        total_input_tokens = 0
        total_output_tokens = 0

        response_model: type[BaseLLMResponse]

        if self.config.dynamic_json_schema:
            response_model = build_llm_response_model(selected_attributes)
        else:
            response_model = LLMResponseSchema

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
                all_annotations.extend(annotations)
                present_attribute_ids = {
                    ann.attribute.attribute_id for ann in annotations
                }
                leaves = [
                    child
                    for concept in leaves
                    if concept.attribute.attribute_id in present_attribute_ids
                    for child in scheme.narrower(concept.identifier)
                ]
                level += 1

        return DocumentExtractionResult(
            annotations=all_annotations,
            messages=all_messages,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=self.model,
        )
