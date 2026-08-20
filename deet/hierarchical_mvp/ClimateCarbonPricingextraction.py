"""
DSPy signatures and extraction pipeline for ClimateCarbonPricing outcome data extraction.

Pipeline steps:
  1. ExtractStudyInfo         — extract study-level metadata and identify all carbon pricing interventions
  2. ExtractEffectOutcomes    — extract effect/outcome data per identified intervention
"""

from __future__ import annotations

import dspy

from .ClimateCarbonPricingmodel import (
    Effect_Outcome,
    Intervention,
    Study,
    Study_Characteristics,
)

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class ExtractStudyInfo(dspy.Signature):
    """
    You are a systematic review assistant.

    Given plain text (converted from PDFs to markdown) from one or more documents
    that all describe the SAME study on carbon pricing, extract all study-level
    metadata and characteristics, and identify every distinct carbon pricing
    intervention analysed.

    Report only information that is explicitly stated in the context.
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same carbon pricing study"
    )
    study_characteristics: Study_Characteristics = dspy.OutputField(
        desc="All study-level metadata and characteristics extracted from the source text"
    )
    interventions: list[Intervention] = dspy.OutputField(
        desc=(
            "Every carbon pricing intervention identified in the study. "
            "Each entry must have an intervention_name."
        )
    )


class ExtractEffectOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same carbon pricing study context and the already-identified
    interventions, extract ALL effect/outcome data reported in the text describing
    the estimated effect of carbon pricing on emissions.

    For EVERY effect outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same carbon pricing study"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="The carbon pricing interventions already identified for this study"
    )
    effect_outcomes: list[Effect_Outcome] = dspy.OutputField(
        desc=("All effect/outcome data reported in the study.")
    )


# ---------------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------------


class ClimateCarbonPricingExtractionPipeline(dspy.Module):
    """
    Two-step DSPy pipeline for structured ClimateCarbonPricing data extraction.

    Step 1 — extract study metadata and identify carbon pricing interventions.
    Step 2 — extract all effect outcomes, informed by the identified interventions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extract_study_info = dspy.Predict(ExtractStudyInfo)
        self.extract_effect_outcomes = dspy.Predict(ExtractEffectOutcomes)

    def forward(self, context: str) -> Study:
        # Step 1: study characteristics + intervention groups
        study_pred = self.extract_study_info(context=context)

        # Step 2: effect outcomes — pass identified interventions as context
        effect_pred = self.extract_effect_outcomes(
            context=context,
            interventions=study_pred.interventions,
        )

        return Study(
            study_characteristics=study_pred.study_characteristics,
            interventions=study_pred.interventions,
            effect_outcomes=effect_pred.effect_outcomes,
        )
