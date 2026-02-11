"""
An example script that calls the extract_from_document
function with an example document and attribute.
Used to test llm configuration settings.
"""

from loguru import logger

from deet.data_models.base import Attribute, AttributeType
from deet.extractors.llm_data_extractor import DataExtractionConfig, LLMDataExtractor
from deet.processors.eppi_annotation_converter import (
    EppiAnnotationConverter,
)
from deet.processors.parser import DocumentParser

parser = DocumentParser()
converter = EppiAnnotationConverter()

# NOTE - define your LLM config stuff here. currently all values are default.
config = DataExtractionConfig()

data_extractor = LLMDataExtractor(config=config)


def main() -> None:
    """
    Call extract_from_document with a dummy document and attribute,
    using the environment settings.
    """
    attr = Attribute(
        question_target="Test question",
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Is the document about climate and health? Return a BOOL",
    )
    context = (
        "This is document, extract data from me please. "
        "I am about climate and health"
    )
    response = data_extractor.extract_from_document(
        attributes=[attr],
        payload=context,
        context_type=None,
    )
    logger.info(response)


if __name__ == "__main__":
    main()
