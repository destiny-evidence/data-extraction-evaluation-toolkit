"""
DSPy extraction pipeline for UN programme evaluation reports.

Pipeline steps:
  1. Extract programme characteristics and interventions.
  2. Extract quantitative outcomes using the characteristics and interventions.
  3. Extract qualitative outcomes with the quantitative outcomes as context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from pathlib import Path

from .programme_framework import (
    PROGRAMME_EXTRACTION_INSTRUCTIONS,
    QUALITATIVE_OUTCOME_INSTRUCTIONS,
    QUANTITATIVE_OUTCOME_INSTRUCTIONS,
    clean_outcome_classifications,
    load_indicator_framework,
)
from .ProgrammeModel import (
    Intervention,
    Programme,
    Programme_Characteristics,
    Qualitative_Outcome,
    Quantitative_Outcome,
)


class ExtractProgrammeInfo(dspy.Signature):
    """
    Extract programme metadata and interventions from a UN evaluation report.

    Report only information explicitly stated in the source. Keep distinct
    interventions separate and preserve names used by the report.
    """

    __doc__ = PROGRAMME_EXTRACTION_INSTRUCTIONS

    context: str = dspy.InputField(
        desc="Markdown text from one or more documents describing the same UN programme evaluation"
    )
    programme_characteristics: Programme_Characteristics = dspy.OutputField(
        desc="Programme-level countries, responsible author, and budget"
    )
    interventions: list[Intervention] = dspy.OutputField(
        desc="Every distinct programme intervention identified in the report"
    )


class ExtractQuantitativeOutcomes(dspy.Signature):
    """Extract numerical outcomes and classify them using the supplied CSV."""

    __doc__ = QUANTITATIVE_OUTCOME_INSTRUCTIONS

    context: str = dspy.InputField(desc="The UN programme evaluation report text")
    classification_framework: str = dspy.InputField(
        desc="CSV category IDs, groups, statements and inclusion/exclusion criteria"
    )
    programme_characteristics: Programme_Characteristics = dspy.InputField(
        desc="Programme characteristics identified in the first pass"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="Programme interventions identified in the first pass"
    )
    quantitative_outcomes: list[Quantitative_Outcome] = dspy.OutputField(
        desc="All reported quantitative outcomes linked to their interventions"
    )


class ExtractQualitativeOutcomes(dspy.Signature):
    """Extract qualitative evidence with previously extracted numbers as context."""

    __doc__ = QUALITATIVE_OUTCOME_INSTRUCTIONS

    context: str = dspy.InputField(desc="The programme evaluation report text")
    classification_framework: str = dspy.InputField(
        desc="CSV category IDs, groups, statements and inclusion/exclusion criteria"
    )
    programme_characteristics: Programme_Characteristics = dspy.InputField(
        desc="Programme characteristics identified in the first pass"
    )
    interventions: list[Intervention] = dspy.InputField(
        desc="Programme interventions identified in the first pass"
    )
    quantitative_outcomes: list[Quantitative_Outcome] = dspy.InputField(
        desc=(
            "Previously extracted quantitative outcomes; context, "
            "not qualitative evidence"
        )
    )
    qualitative_outcomes: list[Qualitative_Outcome] = dspy.OutputField(
        desc="All supported qualitative outcomes, including first-hand experiences"
    )


class ProgrammeExtractionPipeline(dspy.Module):
    """Three-pass structured extraction pipeline for programme evaluations."""

    def __init__(self, classification_framework_path: Path | None = None) -> None:
        """Configure extraction with the bundled or an external CSV framework."""
        super().__init__()
        self.classification_framework = load_indicator_framework(
            classification_framework_path
        )
        self.extract_programme_info = dspy.Predict(ExtractProgrammeInfo)
        self.extract_quantitative = dspy.Predict(ExtractQuantitativeOutcomes)
        self.extract_qualitative = dspy.Predict(ExtractQualitativeOutcomes)

    def forward(self, context: str) -> Programme:
        """Extract and validate outcomes before returning the programme."""
        programme_pred = self.extract_programme_info(context=context)

        quantitative_pred = self.extract_quantitative(
            context=context,
            classification_framework=self.classification_framework,
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
        )
        clean_outcome_classifications(
            quantitative_pred.quantitative_outcomes,
            self.classification_framework,
        )
        qualitative_pred = self.extract_qualitative(
            context=context,
            classification_framework=self.classification_framework,
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
            quantitative_outcomes=quantitative_pred.quantitative_outcomes,
        )
        clean_outcome_classifications(
            qualitative_pred.qualitative_outcomes, self.classification_framework
        )

        return Programme(
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
            quantitative_outcomes=quantitative_pred.quantitative_outcomes,
            qualitative_outcomes=qualitative_pred.qualitative_outcomes,
        )
