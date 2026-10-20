"""Pydantic models for UN programme evaluation report extraction."""

from __future__ import annotations

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

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Outcome(BaseModel):
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
    quantitative_outcome_results: str = Field(
        description="Extract all numerical results reported for this specific outcome, preserving values, units, and relevant context as stated. Do not calculate or infer results. Assign 'NR' if no numerical results are reported for this outcome."
    )
    qualitative_outcome_results: str = Field(
        description="Extract all qualitative results reported for this specific outcome, including narrative findings or observed changes as stated. Do not infer results. Assign 'NR' if no qualitative results are reported for this outcome."
    )
    source_paragraph: str = Field(
        default="",
        description=(
            "Brief verbatim quotes supporting the numerical and qualitative "
            "results for this specific outcome, labelled 'Quantitative' and "
            "'Qualitative'. Include a separate quote and its document name, "
            "section heading and page number for each distinct result where "
            "available. Use only locations present in the supplied text; never "
            "invent page numbers or quotes. State 'location not reported' when "
            "no location is available and 'NR' for a result type with no "
            "reported evidence."
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
    enabler: str = Field(
        description="Enablers or facilitating factors supporting achievement of this outcome. Leave blank if not reported."
    )
    lessons_learned: str = Field(
        description="Lessons learned in relation to this outcome, reported without extrapolation."
    )
    achievements: str = Field(
        description="Achievements associated with this outcome, including reported progress or successes."
    )
    recommendation: str = Field(
        description="Recommendation associated with this outcome. Report it as stated and leave blank if none is reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Programme(BaseModel):
    programme_characteristics: Programme_Characteristics = Field(
        description="Programme-level metadata extracted from the evaluation report."
    )
    interventions: list[Intervention] = Field(
        description="All distinct programme interventions identified in the report."
    )
    outcomes: list[Outcome] = Field(
        default_factory=list,
        description="All outcomes associated with the identified interventions.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Programme_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.programme_characteristics.to_csv_row()
