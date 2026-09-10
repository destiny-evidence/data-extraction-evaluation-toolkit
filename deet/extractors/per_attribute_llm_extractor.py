"""LLM extractor that issues one call per attribute and concatenates results."""

from pathlib import Path

from deet.data_models.base import Attribute, GoldStandardAnnotation
from deet.data_models.documents import ContextType
from deet.data_models.extraction import DocumentExtractionResult
from deet.extractors.llm_data_extractor import LLMDataExtractor


class PerAttributeLLMExtractor(LLMDataExtractor):
    """
    Run each attribute in its own LLM call, then merge the results.

    The default ``LLMDataExtractor`` bundles every attribute into one prompt and
    one call.
    This is efficient, but can degrade performance, as text from one attribute
    can affect how another attribute is interpreted.
    In some cases, we may wish to process each attribute in a separate
    call.
    This method does this, and concatenates the results.
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
        """Extract one attribute per call and concatenate the results."""
        payload = self._resolve_payload(payload=payload, md_path=md_path)
        selected = self._select_attributes(attributes, filter_attribute_ids)

        annotations: list[GoldStandardAnnotation] = []
        messages: list = []
        input_tokens = output_tokens = 0
        llm_call_seconds = 0.0
        model: str | None = None

        for attribute in selected:
            call = super().extract_from_document(
                attributes=[attribute],
                payload=payload,
                context_type=context_type,
            )
            annotations.append(call.annotations[0])
            messages.extend(call.messages)
            input_tokens += call.input_tokens
            output_tokens += call.output_tokens
            llm_call_seconds += call.llm_call_seconds
            model = model or call.model

        return DocumentExtractionResult(
            annotations=annotations,
            messages=messages,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.model,
            llm_call_seconds=llm_call_seconds,
        )
