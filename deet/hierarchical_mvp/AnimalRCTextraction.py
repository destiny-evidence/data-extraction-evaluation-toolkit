"""
DSPy signatures and extraction pipeline for Animal RCT outcome data extraction.

Pipeline steps:
  1. ExtractStudyInfo             — extract study-level metadata and identify all
                                     induction and assessment intervention subpopulations
  2. ExtractDichotomousOutcomes   — extract binary event outcomes per induction/assessment combination
  3. ExtractContinuousOutcomes    — extract mean/SD outcomes per induction/assessment combination
  4. ExtractOtherOutcomes         — extract all other outcomes per induction/assessment combination
"""

from __future__ import annotations

import dspy

from .AnimalRCTmodel import (
    AssessmentIntervention,
    Continuous_Outcome,
    Dichotomous_Outcome,
    InductionIntervention,
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
    that all describe the SAME animal trial, extract all
    study-level metadata and characteristics, and identify every distinct
    induction intervention (subpopulation created to induce a condition or model)
    and every distinct assessment intervention (treatment/procedure being assessed).

    Report only information that is explicitly stated in the context.
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same animal RCT"
    )
    study_characteristics: Study_Characteristics = dspy.OutputField(
        desc="All study-level metadata and characteristics extracted from the source text"
    )
    induction_interventions: list[InductionIntervention] = dspy.OutputField(
        desc=(
            "Every induction intervention (subpopulation) in the trial. "
            "Each entry must have an intervention_to_induce_name and a description."
        )
    )
    assessment_interventions: list[AssessmentIntervention] = dspy.OutputField(
        desc=(
            "Every assessment intervention in the trial. "
            "Each entry must have an intervention_to_assess_name and a description."
        )
    )


class ExtractDichotomousOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same animal RCT context and the already-identified induction and
    assessment interventions, extract ALL dichotomous (binary event) outcome
    data reported in the text, for every unique combination of induction and
    assessment intervention.

    For EVERY dichotomous outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same animal RCT"
    )
    induction_interventions: list[InductionIntervention] = dspy.InputField(
        desc="The induction interventions (subpopulations) already identified for this trial"
    )
    assessment_interventions: list[AssessmentIntervention] = dspy.InputField(
        desc="The assessment interventions already identified for this trial"
    )
    dichotomous_outcomes: list[Dichotomous_Outcome] = dspy.OutputField(
        desc=("All data related to every dichotomous outcomes reported in the study.")
    )


class ExtractContinuousOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same animal RCT context and the already-identified induction and
    assessment interventions, extract ALL continuous outcome data (mean ± SD)
    reported in the text, for every unique combination of induction and
    assessment intervention.

    For EVERY continuous outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report numbers exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same animal RCT"
    )
    induction_interventions: list[InductionIntervention] = dspy.InputField(
        desc="The induction interventions (subpopulations) already identified for this trial"
    )
    assessment_interventions: list[AssessmentIntervention] = dspy.InputField(
        desc="The assessment interventions already identified for this trial"
    )
    continuous_outcomes: list[Continuous_Outcome] = dspy.OutputField(
        desc=("All data related to every continuous outcomes reported in the study.")
    )


class ExtractOtherOutcomes(dspy.Signature):
    """
    You are a systematic review assistant.

    Given the same animal RCT context and the already-identified induction and
    assessment interventions, extract ALL other (non-dichotomous, non-continuous)
    outcome data reported in the text, for every unique combination of induction
    and assessment intervention. Do not re-extract a dichotomous or continuous outcome unless you can identify new data for it that wasn't extracted in the previous steps.

    For EVERY other outcome found, attempt to extract the attributes that are part of the schema attached to this class.

    Report values exactly as they appear in the source — do not calculate or impute.
    If a value is not reported, use the string "NR".
    """

    context: str = dspy.InputField(
        desc="Concatenated markdown text from one or more parsed PDFs, all describing the same animal RCT"
    )
    induction_interventions: list[InductionIntervention] = dspy.InputField(
        desc="The induction interventions (subpopulations) already identified for this trial"
    )
    assessment_interventions: list[AssessmentIntervention] = dspy.InputField(
        desc="The assessment interventions already identified for this trial"
    )

    dichotomous_outcomes: list[Dichotomous_Outcome] = dspy.InputField(
        desc=("All already extracted data related to dichotomous outcomes reported in the study.")
    )

    continuous_outcomes: list[Continuous_Outcome] = dspy.InputField(
        desc=("All already extracted data related to continuous outcomes reported in the study.")
    )

    flexible_outcomes: list[Other_Outcome] = dspy.OutputField(
        desc=("All data related to every 'other type' outcomes reported in the study.")
    )


# ---------------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------------


class AnimalRCTExtractionPipeline(dspy.Module):
    """
    Four-step DSPy pipeline for structured Animal RCT data extraction.

    Step 1 — extract study metadata and identify induction/assessment intervention groups.
    Step 2 — extract all dichotomous outcomes, informed by the identified groups.
    Step 3 — extract all continuous outcomes, informed by the identified groups.
    Step 4 — extract all other outcomes, informed by the identified groups.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extract_study_info = dspy.Predict(ExtractStudyInfo)
        self.extract_dichotomous = dspy.Predict(ExtractDichotomousOutcomes)
        self.extract_continuous = dspy.Predict(ExtractContinuousOutcomes)
        self.extract_other = dspy.Predict(ExtractOtherOutcomes)

    def forward(self, context: str) -> Study:
        # Step 1: study characteristics + induction/assessment intervention groups
        study_pred = self.extract_study_info(context=context)

        # Step 2: dichotomous outcomes — pass identified interventions as context
        dichot_pred = self.extract_dichotomous(
            context=context,
            induction_interventions=study_pred.induction_interventions,
            assessment_interventions=study_pred.assessment_interventions,
        )

        # Step 3: continuous outcomes — pass identified interventions as context
        cont_pred = self.extract_continuous(
            context=context,
            induction_interventions=study_pred.induction_interventions,
            assessment_interventions=study_pred.assessment_interventions,
        )

        # Step 4: other outcomes — pass identified interventions as context
        other_pred = self.extract_other(
            context=context,
            induction_interventions=study_pred.induction_interventions,
            assessment_interventions=study_pred.assessment_interventions,
            dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
            continuous_outcomes=cont_pred.continuous_outcomes,
        )

        return Study(
            study_characteristics=study_pred.study_characteristics,
            induction_interventions=study_pred.induction_interventions,
            assessment_interventions=study_pred.assessment_interventions,
            dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
            continuous_outcomes=cont_pred.continuous_outcomes,
            other_outcomes=other_pred.flexible_outcomes,
        )
