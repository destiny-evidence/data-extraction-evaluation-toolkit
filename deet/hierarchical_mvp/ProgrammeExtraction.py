"""
DSPy extraction pipeline for UN programme evaluation reports.

Pipeline steps:
  1. Extract programme characteristics and interventions.
  2. Extract outcomes, results, and associated learning using the characteristics
     and interventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

if TYPE_CHECKING:
    from pathlib import Path

from .programme_framework import (
    OUTCOME_EXTRACTION_INSTRUCTIONS,
    PROGRAMME_EXTRACTION_INSTRUCTIONS,
    load_indicator_framework,
    validate_classifications,
    validate_sdg_categories,
)
from .ProgrammeModel import (
    Intervention,
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


class ExtractOutcomes(dspy.Signature):
    """Extract source-backed outcomes and classify them using the supplied CSV."""

    __doc__ = OUTCOME_EXTRACTION_INSTRUCTIONS

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
    outcomes: list[Outcome] = dspy.OutputField(
        desc="All reported outcomes linked to their corresponding interventions"
    )


class ProgrammeExtractionPipeline(dspy.Module):
    """Two-pass structured extraction pipeline for UN programme evaluations."""

    def __init__(self, classification_framework_path: Path | None = None) -> None:
        """Configure extraction with the bundled or an external CSV framework."""
        super().__init__()
        self.classification_framework = load_indicator_framework(
            classification_framework_path
        )
        self.extract_programme_info = dspy.Predict(ExtractProgrammeInfo)
        self.extract_outcomes = dspy.Predict(ExtractOutcomes)

    def forward(self, context: str) -> Programme:
        """Extract and validate outcomes before returning the programme."""
        programme_pred = self.extract_programme_info(context=context)

        outcomes_pred = self.extract_outcomes(
            context=context,
            classification_framework=self.classification_framework,
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
        )
        classifications = validate_classifications(
            (
                outcome.outcome_category_classification
                for outcome in outcomes_pred.outcomes
            ),
            self.classification_framework,
        )
        sdg_categories = validate_sdg_categories(
            outcome.SDG_categories for outcome in outcomes_pred.outcomes
        )
        for outcome, classification, sdgs in zip(
            outcomes_pred.outcomes, classifications, sdg_categories, strict=True
        ):
            outcome.outcome_category_classification = classification
            outcome.SDG_categories = sdgs

        return Programme(
            programme_characteristics=programme_pred.programme_characteristics,
            interventions=programme_pred.interventions,
            outcomes=outcomes_pred.outcomes,
        )
