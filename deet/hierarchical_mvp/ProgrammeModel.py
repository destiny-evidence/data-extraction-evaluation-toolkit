"""Pydantic models for UN programme evaluation report extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Programme_Characteristics(BaseModel):
    countries: str = Field(
        description="Countries in which the programme was implemented or evaluated. Report as written and leave blank if not reported."
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
    name: str = Field(description="Name of the programme intervention or activity.")
    description: str = Field(
        description="Description of what the intervention did, whom it targeted, and how it was delivered."
    )
    time_frame: str = Field(
        description="Implementation dates, duration, or other time frame for this intervention. Report as written and leave blank if not reported."
    )
    resources_used: str = Field(
        description="Financial, human, material, or institutional resources used for this intervention. Leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Outcome(BaseModel):
    name: str = Field(description="Name of the reported programme outcome.")
    description: str = Field(
        description="Description of the outcome, including what changed or was achieved and any reported evidence."
    )
    intervention_name: str = Field(
        description="Exact name of the previously identified intervention for which this outcome was measured."
    )
    unit_of_analysis: str = Field(
        description="Unit for which the outcome was analysed or assessed, such as people, households, institutions, communities, or countries. Leave blank if not reported."
    )
    sdg_belonging: str = Field(
        description="UN Sustainable Development Goal or goals to which the outcome belongs. Extract explicit links only and leave blank if not reported."
    )
    strategic_plan_belonging: str = Field(
        description="Strategic plan, result area, output, or objective to which the outcome belongs. Extract explicit links only and leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return list(cls.model_fields)

    def to_csv_row(self) -> dict[str, str]:
        return self.model_dump()


class Learning(BaseModel):
    outcome_name: str = Field(
        description="Exact name of the previously identified outcome to which this learning belongs."
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
    learnings: list[Learning] = Field(
        default_factory=list,
        description="All learning, achievements, and recommendations associated with identified outcomes.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Programme_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.programme_characteristics.to_csv_row()
