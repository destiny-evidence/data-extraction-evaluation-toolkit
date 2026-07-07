"""
DSPy signatures and extraction pipeline for Cochrane RCT data extraction.

Identical pipeline structure to RCTextraction.py but bound to CochraneRCTmodel
classes with Cochrane-specific Study_Characteristics and Intervention fields.

Pipeline steps:
  1. ExtractStudyInfo             — extract study-level metadata and identify all intervention arms
  2. ExtractDichotomousOutcomes   — extract binary event outcomes per identified arm
  3. ExtractContinuousOutcomes    — extract mean/SD outcomes per identified arm
  4. ExtractOtherOutcomes         — extract all remaining outcome types per identified arm
"""

from __future__ import annotations

import dspy

from .CochraneRCTmodel import (
    Continuous_Outcome,
    Dichotomous_Outcome,
    Intervention,
    Other_Outcome,
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
    that all describe the SAME randomized controlled trial, extract all Cochrane-style
    study-level metadata and characteristics, and identify every distinct intervention
    arm.

    Report only information that is explicitly stated in the context.
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same RCT"
    )
    study_characteristics: Study_Characteristics = dspy.OutputField(
        desc="All Cochrane study-level metadata and characteristics extracted from the source text"
    )
    interventions: list[Intervention] = dspy.OutputField(
        desc=(
            "Every intervention arm in the trial. "
            "Each entry must have an arm name, description, and intervention details."
        )
    )


class ExtractDichotomousOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same RCT context and the already-identified intervention arms,
    extract ALL dichotomous (binary event) outcome data reported in the text.

    For EVERY dichotomous outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same RCT"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="The intervention arms already identified for this trial"
    )
    dichotomous_outcomes: list[Dichotomous_Outcome] = dspy.OutputField(
        desc="All data related to every dichotomous outcome reported in the study."
    )


class ExtractContinuousOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same RCT context and the already-identified intervention arms,
    extract ALL continuous outcome data (mean ± SD) reported in the text.

    For EVERY continuous outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same RCT"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="The intervention arms already identified for this trial"
    )
    continuous_outcomes: list[Continuous_Outcome] = dspy.OutputField(
        desc="All data related to every continuous outcome reported in the study."
    )


class ExtractOtherOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same RCT context and the already-identified intervention arms,
    extract ALL other (non-dichotomous, non-continuous) outcome data reported in the text.

    For EVERY other outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report values exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same RCT"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="The intervention arms already identified for this trial"
    )
    flexible_outcomes: list[Other_Outcome] = dspy.OutputField(
        desc="All data related to every 'other type' outcome reported in the study."
    )


# ---------------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------------


class CochraneRCTExtractionPipeline(dspy.Module):
    """
    Four-step DSPy pipeline for structured Cochrane RCT data extraction.

    Step 1 — extract Cochrane study metadata and identify intervention arms.
    Step 2 — extract all dichotomous outcomes, informed by the identified arms.
    Step 3 — extract all continuous outcomes, informed by the identified arms.
    Step 4 — extract all other outcomes, informed by the identified arms.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extract_study_info = dspy.Predict(ExtractStudyInfo)
        self.extract_dichotomous = dspy.Predict(ExtractDichotomousOutcomes)
        self.extract_continuous = dspy.Predict(ExtractContinuousOutcomes)
        self.extract_other = dspy.Predict(ExtractOtherOutcomes)

    def forward(self, context: str) -> Study:
        # Step 1: study characteristics + intervention arms
        study_pred = self.extract_study_info(context=context)

        # Step 2: dichotomous outcomes — pass identified interventions as context
        dichot_pred = self.extract_dichotomous(
            context=context,
            interventions=study_pred.interventions,
        )

        # Step 3: continuous outcomes — pass identified interventions as context
        cont_pred = self.extract_continuous(
            context=context,
            interventions=study_pred.interventions,
        )

        # Step 4: other outcomes — pass identified interventions as context
        other_pred = self.extract_other(
            context=context,
            interventions=study_pred.interventions,
        )

        return Study(
            study_characteristics=study_pred.study_characteristics,
            interventions=study_pred.interventions,
            dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
            continuous_outcomes=cont_pred.continuous_outcomes,
            other_outcomes=other_pred.flexible_outcomes,
        )
