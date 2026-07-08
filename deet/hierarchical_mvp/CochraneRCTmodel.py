"""Pydantic data models for Cochrane RCT data extraction.

Study_Characteristics and Intervention use Cochrane-specific fields.
Outcome models are structured to produce arm-level rows matching the
Study+results CSV template (Study, Outcome, Data type, Arm, Sample size, …).
OtherOutcomes are not used in this extraction variant.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
            "Name or label of the intervention arm "
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
# Outcomes — arm-level fields matching Study+results CSV template
# Columns marked "Contrast level results only" in the template are omitted.
# Each outcome object holds data for both arms and emits two CSV rows.
# ---------------------------------------------------------------------------

# Column name constants matching the CSV template exactly
_ARM_LEVEL_SHARED: list[str] = ["Study", "Outcome", "Data type", "Arm", "Sample size"]
_DICHOTOMOUS_EXTRA: list[str] = ["Cases", "P value", "Footnotes"]
_CONTINUOUS_EXTRA: list[str] = [
    "Mean", "SD", "SE", "Variance",
    "CI level", "CI start", "CI end",
    "t-test", "P value", "Footnotes",
]


class Dichotomous_Outcome(BaseModel):
    """One dichotomous outcome with arm-level data for both arms.

    Produces two CSV rows (one per arm) whose column names match the
    Study+results CSV template.
    """

    outcome_name: str = Field(
        description=(
            "The name of the outcome (maps to the 'Outcome' column in RevMan). "
            "Use the exact name as reported in the study. Must be unique within the study."
        )
    )
    arm_a: str = Field(
        description="Name of the first arm (e.g. the intervention group) exactly as identified in the study."
    )
    arm_b: str = Field(
        description="Name of the second arm (e.g. the control/comparator group) exactly as identified in the study."
    )
    sample_size_a: str = Field(
        description=(
            "Number of participants analysed in arm A for this outcome "
            "('Sample size' column). Leave blank if not reported."
        )
    )
    sample_size_b: str = Field(
        description=(
            "Number of participants analysed in arm B for this outcome "
            "('Sample size' column). Leave blank if not reported."
        )
    )
    cases_a: str = Field(
        description=(
            "Number of cases (events) in arm A — labeled 'events' in RevMan "
            "('Cases' column). Arm-level, dichotomous outcomes only. Leave blank if not reported."
        )
    )
    cases_b: str = Field(
        description=(
            "Number of cases (events) in arm B — labeled 'events' in RevMan "
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

    def to_csv_rows(self, study: str = "") -> list[dict[str, str]]:
        """Return two dicts — one row per arm — ready for csv.DictWriter."""
        base = {"Study": study, "Outcome": self.outcome_name, "Data type": "Arm level"}
        return [
            {**base, "Arm": self.arm_a, "Sample size": self.sample_size_a,
             "Cases": self.cases_a, "P value": self.p_value, "Footnotes": self.footnotes},
            {**base, "Arm": self.arm_b, "Sample size": self.sample_size_b,
             "Cases": self.cases_b, "P value": "", "Footnotes": ""},
        ]


class Continuous_Outcome(BaseModel):
    """One continuous outcome with arm-level data for both arms.

    Produces two CSV rows (one per arm) whose column names match the
    Study+results CSV template.
    """

    outcome_name: str = Field(
        description=(
            "The name of the outcome (maps to the 'Outcome' column in RevMan). "
            "Use the exact name as reported in the study. Must be unique within the study."
        )
    )
    arm_a: str = Field(
        description="Name of the first arm (e.g. the intervention group) exactly as identified in the study."
    )
    arm_b: str = Field(
        description="Name of the second arm (e.g. the control/comparator group) exactly as identified in the study."
    )
    sample_size_a: str = Field(
        description="Number of participants analysed in arm A for this outcome. Leave blank if not reported."
    )
    sample_size_b: str = Field(
        description="Number of participants analysed in arm B for this outcome. Leave blank if not reported."
    )
    mean_a: str = Field(
        description="Mean effect size in arm A ('Mean' column). Leave blank if not reported."
    )
    mean_b: str = Field(
        description="Mean effect size in arm B ('Mean' column). Leave blank if not reported."
    )
    sd_a: str = Field(
        description="Standard deviation of the effect size in arm A ('SD' column). Leave blank if not reported."
    )
    sd_b: str = Field(
        description="Standard deviation of the effect size in arm B ('SD' column). Leave blank if not reported."
    )
    se_a: str = Field(
        default="",
        description="Standard error of the effect size in arm A ('SE' column). Leave blank if not reported.",
    )
    se_b: str = Field(
        default="",
        description="Standard error of the effect size in arm B ('SE' column). Leave blank if not reported.",
    )
    variance_a: str = Field(
        default="",
        description="Variance of the effect size in arm A ('Variance' column). Leave blank if not reported.",
    )
    variance_b: str = Field(
        default="",
        description="Variance of the effect size in arm B ('Variance' column). Leave blank if not reported.",
    )
    ci_level_a: str = Field(
        default="",
        description="Confidence level for the CI in arm A, e.g. 0.95 ('CI level' column). Leave blank if not reported.",
    )
    ci_level_b: str = Field(
        default="",
        description="Confidence level for the CI in arm B, e.g. 0.95 ('CI level' column). Leave blank if not reported.",
    )
    ci_start_a: str = Field(
        default="",
        description="Lower bound of the confidence interval in arm A ('CI start' column). Leave blank if not reported.",
    )
    ci_start_b: str = Field(
        default="",
        description="Lower bound of the confidence interval in arm B ('CI start' column). Leave blank if not reported.",
    )
    ci_end_a: str = Field(
        default="",
        description="Upper bound of the confidence interval in arm A ('CI end' column). Leave blank if not reported.",
    )
    ci_end_b: str = Field(
        default="",
        description="Upper bound of the confidence interval in arm B ('CI end' column). Leave blank if not reported.",
    )
    t_test_a: str = Field(
        default="",
        description="Student-T test statistic for arm A ('t-test' column). Leave blank if not reported.",
    )
    t_test_b: str = Field(
        default="",
        description="Student-T test statistic for arm B ('t-test' column). Leave blank if not reported.",
    )
    p_value_a: str = Field(
        default="",
        description="P-value for arm A ('P value' column). Leave blank if not reported.",
    )
    p_value_b: str = Field(
        default="",
        description="P-value for arm B ('P value' column). Leave blank if not reported.",
    )
    footnotes: str = Field(
        default="",
        description="Any additional notes or context for this outcome (e.g. follow-up time point).",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        """Exact column names from the Study+results CSV template (arm-level, continuous)."""
        return _ARM_LEVEL_SHARED + _CONTINUOUS_EXTRA

    def to_csv_rows(self, study: str = "") -> list[dict[str, str]]:
        """Return two dicts — one row per arm — ready for csv.DictWriter."""
        base = {"Study": study, "Outcome": self.outcome_name, "Data type": "Arm level"}
        return [
            {
                **base,
                "Arm": self.arm_a, "Sample size": self.sample_size_a,
                "Mean": self.mean_a, "SD": self.sd_a, "SE": self.se_a,
                "Variance": self.variance_a, "CI level": self.ci_level_a,
                "CI start": self.ci_start_a, "CI end": self.ci_end_a,
                "t-test": self.t_test_a, "P value": self.p_value_a,
                "Footnotes": self.footnotes,
            },
            {
                **base,
                "Arm": self.arm_b, "Sample size": self.sample_size_b,
                "Mean": self.mean_b, "SD": self.sd_b, "SE": self.se_b,
                "Variance": self.variance_b, "CI level": self.ci_level_b,
                "CI start": self.ci_start_b, "CI end": self.ci_end_b,
                "t-test": self.t_test_b, "P value": self.p_value_b,
                "Footnotes": "",
            },
        ]


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

