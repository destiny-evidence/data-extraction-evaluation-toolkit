"""Simple LLM evaluation - just the essentials."""

import json
import os
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.logger import logger

# Load environment variables
load_dotenv(override=True)


class AttributeAnswerCoT(BaseModel):
    """Detailed answer format for a single attribute with reasoning."""

    attribute_name: str = Field(
        description="The name of the attribute being asked about"
    )
    Answer: str = Field(description="The answer to the question, 'True' or 'False'")
    Reasoning: str = Field(description="The reasoning behind the answer")
    Citation: str | None = Field(
        description="The citation from the Research Information to support the answer"
    )


class BatchAnswerFormatCoT(BaseModel):
    """Batch answers for all attributes with reasoning."""

    answers: list[AttributeAnswerCoT] = Field(
        description="List of answers for each attribute"
    )


def call_llm(
    document_context: str, attributes: list[dict[str, Any]]
) -> BatchAnswerFormatCoT:
    """Call LLM to evaluate document against attributes."""
    # Create prompt
    attributes_text = "\n".join(
        [
            f"- {attr['attribute_label']} (ID: {attr['attribute_id']}): {attr['attribute_set_description']}"
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
    "answers": [
        {{
            "attribute_name": "Attribute Name",
            "Answer": "True" or "False",
            "Reasoning": "Your reasoning",
            "Citation": "Supporting text from document (or null)"
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
    return BatchAnswerFormatCoT(**result)


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
        attributes = json.load(f)

    # Get first document and first 2 attributes
    sample_doc = annotated_docs[0]
    sample_attributes = attributes[:2]

    # Extract document context
    document_context = sample_doc.get("document", {}).get("abstract", "")
    if not document_context:
        document_context = sample_doc.get("document", {}).get("title", "")

    # Prepare attributes for LLM
    attrs = [
        {
            "attribute_id": attr["attribute_id"],
            "attribute_label": attr["attribute_label"],
            "attribute_set_description": attr["attribute_set_description"],
        }
        for attr in sample_attributes
    ]

    logger.info("Calling LLM...")
    result = call_llm(document_context, attrs)

    # Display results
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    for answer in result.answers:
        logger.info("")
        logger.info(answer.attribute_name)
        logger.info(f"Answer: {answer.Answer}")
        logger.info(f"Reasoning: {answer.Reasoning}")
        if answer.Citation:
            logger.info(f"Citation: {answer.Citation}")
        logger.info("-" * 40)

    # Save results
    output_file = base_dir / "simple_evaluation_results.json"
    with output_file.open("w") as f:
        json.dump(result.model_dump(), f, indent=2)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
