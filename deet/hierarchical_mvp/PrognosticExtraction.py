"""DSPy signatures and extraction pipeline for prognostic study data extraction.

Pipeline steps:
  1. ExtractPrognosticStudyInfo    — extract study-level metadata and identify all prognostic factors
  2. ExtractHazardRatioOutcomes    — extract hazard ratio outcomes, informed by identified factors
  3. ExtractOtherPrognosticOutcomes — extract all other prognostic outcomes, informed by identified factors
"""

from __future__ import annotations

import dspy

from .PrognosticModel import (
    HazardRatioOutcome,
    OtherPrognosticOutcome,
    PrognosticFactor,
    PrognosticStudy,
    PrognosticStudy_Characteristics,
)

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class ExtractPrognosticStudyInfo(dspy.Signature):
    """
    You are a systematic review assistant.

    Given plain text (converted from PDFs to markdown) from one or more documents
    that all describe the SAME prognostic study, extract all study-level metadata
    and characteristics, and identify every distinct prognostic factor examined.

    Report only information that is explicitly stated in the context.
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same prognostic study"
    )
    study_characteristics: PrognosticStudy_Characteristics = dspy.OutputField(
        desc="All study-level metadata and characteristics extracted from the source text"
    )
    prognostic_factors: list[PrognosticFactor] = dspy.OutputField(
        desc=(
            "Every prognostic factor examined in the study. "
            "Each entry must have a factor_name and a description."
        )
    )


class ExtractHazardRatioOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same prognostic study context and the already-identified prognostic factors,
    extract ALL outcomes reported as hazard ratios.

    For EVERY hazard ratio outcome found, attempt to extract all attributes in the schema.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same prognostic study"
    )
    prognostic_factors: list[PrognosticFactor] = dspy.InputField(
        desc="The prognostic factors already identified for this study"
    )
    hazard_ratio_outcomes: list[HazardRatioOutcome] = dspy.OutputField(
        desc="All outcomes reported as hazard ratios in the study."
    )


class ExtractOtherPrognosticOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same prognostic study context and the already-identified prognostic factors,
    extract ALL outcomes NOT reported as hazard ratios (e.g. response rates, means, medians,
    odds ratios, percentages).

    For EVERY other prognostic outcome found, attempt to extract all attributes in the schema.

    Report values exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same prognostic study"
    )
    prognostic_factors: list[PrognosticFactor] = dspy.InputField(
        desc="The prognostic factors already identified for this study"
    )
    other_prognostic_outcomes: list[OtherPrognosticOutcome] = dspy.OutputField(
        desc="All non-hazard-ratio prognostic outcomes reported in the study."
    )


# ---------------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------------


class PrognosticExtractionPipeline(dspy.Module):
    """
    Three-step DSPy pipeline for structured prognostic study data extraction.

    Step 1 — extract study metadata and identify prognostic factors.
    Step 2 — extract all hazard ratio outcomes, informed by identified factors.
    Step 3 — extract all other prognostic outcomes, informed by identified factors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extract_study_info = dspy.Predict(ExtractPrognosticStudyInfo)
        self.extract_hazard_ratio = dspy.Predict(ExtractHazardRatioOutcomes)
        self.extract_other = dspy.Predict(ExtractOtherPrognosticOutcomes)

    def forward(self, context: str) -> PrognosticStudy:
        # Step 1: study characteristics + prognostic factors
        study_pred = self.extract_study_info(context=context)

        # Step 2: hazard ratio outcomes — pass identified factors as context
        hr_pred = self.extract_hazard_ratio(
            context=context,
            prognostic_factors=study_pred.prognostic_factors,
        )

        # Step 3: other prognostic outcomes — pass identified factors as context
        other_pred = self.extract_other(
            context=context,
            prognostic_factors=study_pred.prognostic_factors,
        )

        return PrognosticStudy(
            study_characteristics=study_pred.study_characteristics,
            prognostic_factors=study_pred.prognostic_factors,
            hazard_ratio_outcomes=hr_pred.hazard_ratio_outcomes,
            other_prognostic_outcomes=other_pred.other_prognostic_outcomes,
        )
