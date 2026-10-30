"""Pydantic models for UN programme evaluation report extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .programme_framework import SDG_CATEGORIES


class Programme_Characteristics(BaseModel):
    countries: str = Field(
        description="Countries in which the programme was implemented or evaluated, separated by semicolon (;). Report as written and leave blank if not reported."
    )
    responsible_author: str = Field(
        description="Author, organisation, office, or other entity responsible for the programme evaluation report. Leave blank if not reported."
    )
    budget: str = Field(
        description="Programme budget or expenditure, including currency, period, and whether the amount was planned or spent when reported. Do not calculate or infer values."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Intervention(BaseModel):
    name: str = Field(description="Name of the programme intervention, as reported in the evaluation document.")
    description: str = Field(
        description="Description of what the intervention did, whom it targeted, and how it was delivered."
    )
    time_frame: str = Field(
        description="Implementation dates, duration, or other time frame for this intervention. Report as written and leave blank if not reported."
    )
    resources_used: str = Field(
        description="Financial, human, material, or institutional resources used for this intervention. Leave blank if not reported."
    )
    intended_results: str = Field(
        default="",
        description=(
            "The explicitly stated intended outcomes, aims, ideal results or "
            "targets for this intervention, regardless of actual results. "
            "Include target values and time frames where reported. Separately "
            "describe how reported actual results differed from the stated aims "
            "only when the source supports that comparison; preserve reported "
            "values without calculating differences or inferring causal "
            "explanations. State when aims, actual results or comparisons are "
            "not reported; do not invent ideal outcomes."
        ),
    )
    source_grounding: str = Field(
        default="",
        description=(
            "Brief verbatim quotes supporting the reported details of this "
            "intervention. Include the chapter name, section heading and page "
            "number where available. Use only locations present in the supplied "
            "text; never invent quotes or locations. Leave blank if no supporting "
            "source text is available. State 'location not reported' if no "
            "location is available."
        ),
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class _OutcomeBase(BaseModel):
    outcome_type: str = Field(
        description="Whether the outcome is quantitative or qualitative."
    )
    name: str = Field(description="Name of the reported programme outcome.")
    description: str = Field(
        description="Description of the outcome, including how it was measured."
    )
    intervention_name: str = Field(
        description="Exact name of the previously identified intervention for which this outcome was measured."
    )
    unit_of_analysis: str = Field(
        description="Unit for which the outcome was analysed or assessed, such as people, households, institutions, communities, or countries. Leave blank if not reported."
    )
    SDG_categories: str = Field(
        default="",
        description=(
            "A semicolon-separated list of all fitting UN Sustainable Development "
            "Goals based on the reported outcome evidence. Analytical assignment "
            "is allowed even without an explicit SDG link in the report. Copy "
            "only exact labels from this list, without numbers, prefixes or "
            "duplicates, and leave blank if none applies: "
            + "; ".join(SDG_CATEGORIES)
        ),
    )
    results_data_source: str = Field(
        default="",
        description=(
            "Brief verbatim quotes supporting the results for this specific "
            "outcome. Include the chapter name, section heading and page number "
            "where available. Use only locations present in the supplied text; "
            "never invent page numbers, chapter or section names, or quotes. "
            "Leave blank if no supporting source text is available. State "
            "'location not reported' if no location is available."
        )
    )
    outcome_category_classification: str = Field(
        default="",
        description=(
            "All applicable categories from the supplied CSV classification "
            "framework as a semicolon-separated string. Each entry must be "
            "exactly '<category_id>: <category_statement>', copying the ID and "
            "full statement verbatim. For indicator frameworks these are the "
            "indicator ID and indicator statement. Classify the reported outcome "
            "evidence using inclusion_criteria and exclusion_criteria, not merely "
            "keywords or intended aims. Multiple categories are allowed only "
            "when independently supported; do not repeat entries. This is "
            "analytical classification, not an assertion that the report "
            "explicitly named the category. Leave blank if no category is "
            "supported."
        )
    )

    barrier: str = Field(
        description="Barriers or constraints affecting achievement of this outcome. Leave blank if not reported."
    )
    barrier_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this barrier. Include the chapter "
            "name, section heading and page number where available. Do not invent "
            "quotes or locations; leave blank if no supporting source text is "
            "available. State 'location not reported' if no location is available."
        ),
    )
    enabler: str = Field(
        description="Enablers or facilitating factors supporting achievement of this outcome. Leave blank if not reported."
    )
    enabler_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this enabler. Include the chapter "
            "name, section heading and page number where available. Do not invent "
            "quotes or locations; leave blank if no supporting source text is "
            "available. State 'location not reported' if no location is available."
        ),
    )
    lessons_learned: str = Field(
        description="Lessons learned in relation to this outcome, reported without extrapolation."
    )
    lessons_learned_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this lesson learned. Include the "
            "chapter name, section heading and page number where available. Do "
            "not invent quotes or locations; leave blank if no supporting source "
            "text is available. State 'location not reported' if no location is "
            "available."
        ),
    )
    achievements: str = Field(
        description="Achievements associated with this outcome, including reported progress or successes."
    )
    achievements_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this achievement. Include the "
            "chapter name, section heading and page number where available. Do "
            "not invent quotes or locations; leave blank if no supporting source "
            "text is available. State 'location not reported' if no location is "
            "available."
        ),
    )
    recommendation: str = Field(
        description="Recommendation associated with this outcome. Report it as stated and leave blank if none is reported."
    )
    recommendation_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this recommendation. Include the "
            "chapter name, section heading and page number where available. Do "
            "not invent quotes or locations; leave blank if no supporting source "
            "text is available. State 'location not reported' if no location is "
            "available."
        ),
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Quantitative_Outcome(_OutcomeBase):
    """An outcome supported by reported numerical results."""

    outcome_type: Literal["quantitative"] = Field(
        default="quantitative", description="Always 'quantitative'."
    )
    baseline_data: str = Field(
        default="",
        description=(
            "Relevant reported quantitative or qualitative baseline information for "
            "this specific outcome: the status quo before the intervention was "
            "implemented. Preserve reported values, units, dates, population "
            "and relevant context. Include narrative descriptions of initial "
            "conditions where reported. Do not confuse targets or post-intervention "
            "results with baseline data, calculate baseline values, or infer "
            "baseline conditions from later changes. Assign 'NR' if not reported."
        ),
    )
    baseline_data_source: str = Field(
        default="",
        description=(
            "Brief verbatim quote supporting this baseline data. Include the "
            "chapter name, section heading and page number where available. Do "
            "not invent quotes or locations; leave blank if no supporting source "
            "text is available. State 'location not reported' if no location is "
            "available."
        ),
    )
    results: str = Field(
        description=(
            "All numerical results reported for this specific outcome, preserving "
            "values, units, time points, population and relevant context as stated. "
            "Do not calculate or infer results. Do not create a quantitative "
            "outcome when no numerical result is reported."
        )
    )


class Qualitative_Outcome(_OutcomeBase):
    """An outcome supported by qualitative findings or first-hand experiences."""

    outcome_type: Literal["qualitative"] = Field(
        default="qualitative", description="Always 'qualitative'."
    )
    results: str = Field(
        description=(
            "Reported qualitative findings or observed changes for this specific "
            "outcome, including first-hand experiences from interviews, focus "
            "groups, free-text responses and participant or beneficiary feedback. "
            "Preserve attribution, differing views and relevant context. "
            "Do not infer findings or merely rephrase numerical results. "
            "Do not create a qualitative outcome without qualitative evidence."
        )
    )


class Programme(BaseModel):
    programme_characteristics: Programme_Characteristics = Field(
        description="Programme-level metadata extracted from the evaluation report."
    )
    interventions: list[Intervention] = Field(
        description="All distinct programme interventions identified in the report."
    )
    quantitative_outcomes: list[Quantitative_Outcome] = Field(
        default_factory=list,
        description="All quantitative outcomes associated with the interventions.",
    )
    qualitative_outcomes: list[Qualitative_Outcome] = Field(
        default_factory=list,
        description="All qualitative outcomes associated with the interventions.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Programme_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.programme_characteristics.to_csv_row()

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        """Return columns for the shared quantitative/qualitative outcomes table."""
        return list(dict.fromkeys(
            Quantitative_Outcome.csv_fieldnames() + Qualitative_Outcome.csv_fieldnames()
        ))
