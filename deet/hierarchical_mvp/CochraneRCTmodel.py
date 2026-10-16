"""Pydantic data models for Cochrane RCT data extraction.

Study_Characteristics and Intervention use Cochrane-specific fields.
Outcome models are structured to produce arm-level rows matching the
Study+results CSV template (Study, Outcome, Data type, Arm, Sample size, …).
OtherOutcomes are not used in this extraction variant.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .continuous_statistics import (
    ReportedStatistic,
    revman_statistics,
    summary_statistics_field,
    variability_statistics_field,
)

# ---------------------------------------------------------------------------
# Study characteristics (Cochrane-specific)
# ---------------------------------------------------------------------------


class Study_Characteristics(BaseModel):
    study: str = Field(
        description=(
            "Unique study identifier in 'STD-first author-year' format "
            "(e.g. 'STD-Smith-2021'). This value must be consistent across "
            "all extracted data rows for this study."
        )
    )
    year: str = Field(
        description="Year of publication or study year."
    )
    data_source: str = Field(
        description="Data source for the study (e.g. database, registry, or publication venue)."
    )
    id_doi: str = Field(
        description="DOI identifier for the study. Leave blank if not reported."
    )
    char_methods: str = Field(
        description=(
            "Cochrane characteristic: Methods. Describe the study design and methods "
            "as reported (e.g. RCT, cluster-RCT, blinding, allocation concealment)."
        )
    )
    char_participants: str = Field(
        description=(
            "Cochrane characteristic: Participants. Describe study participants including "
            "eligibility criteria, setting, and key demographics."
        )
    )
    char_interventions: str = Field(
        description=(
            "Cochrane characteristic: Interventions. Describe all intervention and "
            "control arms, including dose, frequency, and duration where reported."
        )
    )
    char_outcomes: str = Field(
        description=(
            "Cochrane characteristic: Outcomes. List all primary and secondary outcomes "
            "measured, including time points where reported."
        )
    )
    char_notes: str = Field(
        description=(
            "Cochrane characteristic: Notes. Any additional notes or remarks about the "
            "study (e.g. funding, conflicts of interest, related publications)."
        )
    )
    cov_disease_severity: str = Field(
        description=(
            "Covariate: Disease severity. Describe the disease severity of participants "
            "at baseline if reported. Leave blank if not reported."
        )
    )
    cov_treatment_duration: str = Field(
        description=(
            "Covariate: Treatment duration. Provide the duration of the treatment or "
            "intervention period. Leave blank if not reported."
        )
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "year",
            "data_source",
            "id_doi",
            "char_methods",
            "char_participants",
            "char_interventions",
            "char_outcomes",
            "char_notes",
            "cov_disease_severity",
            "cov_treatment_duration",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "study": self.study,
            "year": self.year,
            "data_source": self.data_source,
            "id_doi": self.id_doi,
            "char_methods": self.char_methods,
            "char_participants": self.char_participants,
            "char_interventions": self.char_interventions,
            "char_outcomes": self.char_outcomes,
            "char_notes": self.char_notes,
            "cov_disease_severity": self.cov_disease_severity,
            "cov_treatment_duration": self.cov_treatment_duration,
        }


# ---------------------------------------------------------------------------
# Intervention (Cochrane-specific)
# ---------------------------------------------------------------------------


class Intervention(BaseModel):
    arm: str = Field(
        description=(
            "Identify and extract the name of the experimental groups from this study. "
            "Use the name or label of this group "
            "(e.g. 'Treatment A', 'Control', 'Placebo')."
        )
    )
    description: str = Field(
        description=(
            "Description of this intervention arm, including any relevant details "
            "about the participants or context specific to this arm."
        )
    )
    intervention: str = Field(
        description=(
            "The intervention applied in this arm. Describe the treatment, procedure, "
            "or exposure in detail including dose, frequency, and duration if reported."
        )
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "arm",
            "description",
            "intervention",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "arm": self.arm,
            "description": self.description,
            "intervention": self.intervention,
        }


# ---------------------------------------------------------------------------
# Outcomes — single-arm fields matching the Study+results CSV template.
# Columns marked "Contrast level results only" in the template are omitted.
# ---------------------------------------------------------------------------

# Column name constants matching the CSV template exactly
_ARM_LEVEL_SHARED: list[str] = ["Study", "Outcome", "Data type", "Arm", "Sample size"]
_DICHOTOMOUS_EXTRA: list[str] = ["Cases", "P value", "Footnotes"]
_CONTINUOUS_EXTRA: list[str] = [
    "Mean", "SD", "SE", "Variance",
    "CI level", "CI start", "CI end",
    "t-test", "P value", "Footnotes",
    "group_summary_statistics", "group_variability_statistics",
]


class Dichotomous_Outcome(BaseModel):
    """One dichotomous outcome with data for one group."""

    outcome_name: str = Field(
        description=(
            "The name of the outcome (maps to the 'Outcome' column in RevMan). "
            "Use the exact name as reported in the study. Must be unique within the study."
        )
    )
    group_label: str = Field(
        description="Name of this group (intervention or control) exactly as identified in the study."
    )
    group_N: str = Field(
        description=(
            "Number of participants analysed in this group for this outcome "
            "('Sample size' column). Leave blank if not reported."
        )
    )
    group_Events: str = Field(
        description=(
            "Number of cases (events) in this group — labeled 'events' in RevMan "
            "('Cases' column). Arm-level, dichotomous outcomes only. Leave blank if not reported."
        )
    )
    p_value: str = Field(
        default="",
        description="P-value for this outcome if reported. Leave blank if not reported.",
    )
    footnotes: str = Field(
        default="",
        description="Any additional notes or context for this outcome (e.g. follow-up time point).",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        """Exact column names from the Study+results CSV template (arm-level, dichotomous)."""
        return _ARM_LEVEL_SHARED + _DICHOTOMOUS_EXTRA

    def to_csv_row(self, study: str = "") -> dict[str, str]:
        """Return one arm-level row ready for csv.DictWriter."""
        return {
            "Study": study,
            "Outcome": self.outcome_name,
            "Data type": "Arm level",
            "Arm": self.group_label,
            "Sample size": self.group_N,
            "Cases": self.group_Events,
            "P value": self.p_value,
            "Footnotes": self.footnotes,
        }

    def to_csv_rows(self, study: str = "") -> list[dict[str, str]]:
        """Keep the exporter API while emitting only this arm."""
        return [self.to_csv_row(study=study)]


class Continuous_Outcome(BaseModel):
    """One continuous outcome with data for one group."""

    outcome_name: str = Field(
        description=(
            "The name of the outcome (maps to the 'Outcome' column in RevMan). "
            "Use the exact name as reported in the study. Must be unique within the study."
        )
    )
    group_label: str = Field(
        description="Name of this group (intervention or control) exactly as identified in the study."
    )
    group_N: str = Field(
        description="Number of participants analysed in this group for this outcome. Leave blank if not reported."
    )
    group_summary_statistics: list[ReportedStatistic] = summary_statistics_field(
        "this arm", single=True
    )
    group_variability_statistics: list[ReportedStatistic] = (
        variability_statistics_field("this arm", single=True)
    )
    group_t_test: str = Field(
        default="",
        description="Student-T test statistic for this group ('t-test' column). Leave blank if not reported.",
    )
    group_p_value: str = Field(
        default="",
        description="P-value for this group ('P value' column). Leave blank if not reported.",
    )
    footnotes: str = Field(
        default="",
        description="Any additional notes or context for this outcome (e.g. follow-up time point).",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        """Exact column names from the Study+results CSV template (arm-level, continuous)."""
        return _ARM_LEVEL_SHARED + _CONTINUOUS_EXTRA

    def to_csv_row(self, study: str = "") -> dict[str, str]:
        """Return one arm-level row ready for csv.DictWriter."""
        statistics = revman_statistics(
            self.group_summary_statistics, self.group_variability_statistics
        )
        return {
            "Study": study,
            "Outcome": self.outcome_name,
            "Data type": "Arm level",
            "Arm": self.group_label,
            "Sample size": self.group_N,
            "Mean": statistics["Mean"],
            "SD": statistics["SD"],
            "SE": statistics["SE"],
            "Variance": statistics["Variance"],
            "CI level": "",
            "CI start": "",
            "CI end": "",
            "t-test": self.group_t_test,
            "P value": self.group_p_value,
            "Footnotes": self.footnotes,
            "group_summary_statistics": statistics["group_summary_statistics"],
            "group_variability_statistics": statistics["group_variability_statistics"],
        }

    def to_csv_rows(self, study: str = "") -> list[dict[str, str]]:
        """Keep the exporter API while emitting only this arm."""
        return [self.to_csv_row(study=study)]


# ---------------------------------------------------------------------------
# Top-level Study model
# ---------------------------------------------------------------------------


class Study(BaseModel):
    study_characteristics: Study_Characteristics = Field(
        description="Cochrane study-level metadata and characteristics."
    )
    interventions: list[Intervention] = Field(
        description="All intervention arms identified in the trial."
    )
    dichotomous_outcomes: list[Dichotomous_Outcome] = Field(
        default_factory=list,
        description="All dichotomous (binary event) outcomes extracted from the study.",
    )
    continuous_outcomes: list[Continuous_Outcome] = Field(
        default_factory=list,
        description="All continuous outcomes extracted from the study.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Study_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.study_characteristics.to_csv_row()

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        """Union of dichotomous and continuous arm-level column names, in declaration order."""
        return list(
            dict.fromkeys(
                Dichotomous_Outcome.csv_fieldnames()
                + Continuous_Outcome.csv_fieldnames()
            )
        )
