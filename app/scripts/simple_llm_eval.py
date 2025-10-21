"""Simple LLM evaluation - just the essentials."""

import json
import os
from pathlib import Path

import litellm
from dotenv import load_dotenv

from app.logger import logger
from app.models.base import AnnotationType
from app.models.eppi import EppiAttribute, EppiGoldStandardAnnotation

# Load environment variables
load_dotenv(override=True)


def call_llm(
    document_context: str, attributes: list[EppiAttribute]
) -> list[EppiGoldStandardAnnotation]:
    """Call LLM to evaluate document against attributes."""
    # Create prompt
    attributes_text = "\n".join(
        [
            f"- {attr.attribute_label} (ID: {attr.attribute_id}): {attr.attribute_set_description or 'No description'}"
            for attr in attributes
        ]
    )

    prompt = f"""
Analyze this research document and answer questions about specific attributes. For each attribute, determine if it's present (True/False), provide reasoning, and include citations.

Document:
Title: {document_context.split('.')[0] if '.' in document_context else 'Research Document'}

Abstract: {document_context[:500]}...

Context: {document_context}

Attributes:
{attributes_text}

Respond in JSON format:
{{
    "annotations": [
        {{
            "attribute_id": "attribute_id",
            "output_data": true or false,
            "annotation_type": "llm",
            "additional_text": "Supporting text from document (or null)",
            "reasoning": "Your reasoning for the decision"
        }}
    ]
}}
"""

    # Call LLM
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Use Azure model format if Azure is configured
    if os.getenv("AZURE_API_KEY"):
        deployment = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
        model = f"azure/{deployment}"

    # Prepare messages
    messages = [
        {
            "role": "system",
            "content": "You are an expert research analyst. Provide detailed, accurate analysis.",
        },
        {"role": "user", "content": prompt},
    ]

    # Log the request
    logger.info("=" * 60)
    logger.info("LLM REQUEST")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info("Temperature: 0.1")
    logger.info("Response Format: JSON Object")
    logger.info(f"Provider: {'Azure' if os.getenv('AZURE_API_KEY') else 'OpenAI'}")
    logger.info("")
    logger.info("System Message:")
    logger.info(messages[0]["content"])
    logger.info("")
    logger.info("User Message:")
    logger.info(messages[1]["content"])
    logger.info("=" * 60)

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    # Log the response
    response_content = response.choices[0].message.content
    logger.info("")
    logger.info("LLM RESPONSE")
    logger.info("=" * 60)
    logger.info("Raw Response:")
    logger.info(response_content)
    logger.info("=" * 60)

    # Parse response
    result = json.loads(response_content)

    # Create EppiGoldStandardAnnotation objects
    annotations = []
    for annotation_data in result.get("annotations", []):
        # Find the corresponding attribute
        attribute_id = annotation_data.get("attribute_id")
        attribute = next(
            (attr for attr in attributes if attr.attribute_id == attribute_id), None
        )

        if attribute:
            annotation = EppiGoldStandardAnnotation(
                attribute=attribute,
                output_data=annotation_data.get("output_data", False),
                annotation_type=AnnotationType.LLM,
                additional_text=annotation_data.get("additional_text"),
                reasoning=annotation_data.get("reasoning"),
            )
            annotations.append(annotation)

    return annotations


def main() -> None:
    """Run simple LLM evaluation."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    annotations_file = (
        base_dir / "annotations" / "processed" / "eppi" / "annotated_documents.json"
    )
    attributes_file = (
        base_dir / "annotations" / "processed" / "eppi" / "attributes.json"
    )

    logger.info("Loading document and attributes...")

    # Load data
    with annotations_file.open() as f:
        annotated_docs = json.load(f)

    with attributes_file.open() as f:
        attributes_data = json.load(f)

    # Get first document and first 2 attributes
    sample_doc = annotated_docs[0]
    sample_attributes_data = attributes_data[:2]

    # Extract document context
    document_context = sample_doc.get("context", "")
    if not document_context:
        document_context = sample_doc.get("name", "")

    # Convert attributes data to EppiAttribute objects
    attributes = []
    for attr_data in sample_attributes_data:
        # Convert the JSON data to EppiAttribute format
        processed_attr_data = {
            "question_target": "",
            "output_data_type": bool,
            "attribute_id": str(attr_data.get("attribute_id", "")),
            "attribute_label": attr_data.get("attribute_label", ""),
            "attribute_set_description": attr_data.get("attribute_set_description"),
            "attribute_type": attr_data.get("attribute_type"),
        }
        attribute = EppiAttribute.model_validate(processed_attr_data)
        attributes.append(attribute)

    logger.info("Calling LLM...")
    annotations = call_llm(document_context, attributes)

    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for annotation in annotations:
        logger.info("")
        logger.info(annotation.attribute.attribute_label)
        logger.info(f"Output Data: {annotation.output_data}")
        logger.info(f"Reasoning: {annotation.reasoning}")
        if annotation.additional_text:
            logger.info(f"Additional Text: {annotation.additional_text}")
        logger.info("-" * 40)

    # Save results
    output_file = base_dir / "simple_evaluation_results.json"
    with output_file.open("w") as f:
        # Convert annotations to dict for JSON serialization
        annotations_data = []
        for annotation in annotations:
            annotation_dict = annotation.model_dump()
            # Convert type objects to strings for JSON serialization
            if (
                "attribute" in annotation_dict
                and "output_data_type" in annotation_dict["attribute"]
            ):
                annotation_dict["attribute"]["output_data_type"] = str(
                    annotation_dict["attribute"]["output_data_type"]
                )
            annotations_data.append(annotation_dict)
        json.dump({"annotations": annotations_data}, f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
