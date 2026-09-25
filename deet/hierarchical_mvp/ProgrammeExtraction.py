"""
DSPy extraction pipeline for UN programme evaluation reports.

Pipeline steps:
  1. Extract programme characteristics and interventions.
  2. Extract outcomes using the characteristics and interventions.
  3. Extract learnings using all previously extracted information.
"""

from __future__ import annotations

import dspy

from .ProgrammeModel import (
    Intervention,
    Learning,
    Outcome,
    Programme,
    Programme_Characteristics,
)


class ExtractProgrammeInfo(dspy.Signature):
    """
    Extract programme metadata and interventions from a UN evaluation report.

    Report only information explicitly stated in the source. Keep distinct
    interventions separate and preserve names used by the report.
    """

    context: str = dspy.InputField(
        desc="Markdown text from one or more documents describing the same UN programme evaluation"
    )
    programme_characteristics: Programme_Characteristics = dspy.OutputField(
        desc="Programme-level countries, responsible author, and budget"
    )
    interventions: list[Intervention] = dspy.OutputField(
        desc="Every distinct programme intervention identified in the report"
    )


class ExtractOutcomes(dspy.Signature):
    """
    Extract outcomes for the interventions identified in the first pass.

    Link each outcome to an intervention by its exact extracted name. Report only
    evidence stated in the source and do not infer SDG or strategic-plan links.
    """

    context: str = dspy.InputField(desc="The UN programme evaluation report text")
    programme_characteristics: Programme_Characteristics = dspy.InputField(
        desc="Programme characteristics identified in the first pass"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="Programme interventions identified in the first pass"
    )
    outcomes: list[Outcome] = dspy.OutputField(
        desc="All reported outcomes linked to their corresponding interventions"
    )


class ExtractLearnings(dspy.Signature):
    """
    Extract learning associated with outcomes from a UN evaluation report.

    Use all prior classifications and link each learning to an outcome by its exact
    extracted name. Do not invent links or combine unrelated findings.
    """

    context: str = dspy.InputField(desc="The UN programme evaluation report text")
    programme_characteristics: Programme_Characteristics = dspy.InputField(
        desc="Programme characteristics identified in the first pass"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="Programme interventions identified in the first pass"
    )
    outcomes: list[Outcome] = dspy.InputField(
        desc="Programme outcomes identified in the second pass"
    )
    learnings: list[Learning] = dspy.OutputField(
        desc="Barriers, enablers, lessons, achievements, and recommendations linked to outcomes"
    )


class ProgrammeExtractionPipeline(dspy.Module):
    """Three-pass structured extraction pipeline for UN programme evaluations."""

    def __init__(self) -> None:
        super().__init__()
        self.extract_programme_info = dspy.Predict(ExtractProgrammeInfo)
        self.extract_outcomes = dspy.Predict(ExtractOutcomes)
        self.extract_learnings = dspy.Predict(ExtractLearnings)

    def forward(self, context: str) -> Programme:
        programme_pred = self.extract_programme_info(context=context)

        outcomes_pred = self.extract_outcomes(
            context=context,
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
        )

        learnings_pred = self.extract_learnings(
            context=context,
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
            outcomes=outcomes_pred.outcomes,
        )

        return Programme(
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
            outcomes=outcomes_pred.outcomes,
            learnings=learnings_pred.learnings,
        )
