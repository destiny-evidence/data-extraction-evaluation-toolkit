"""Pydantic data models for Cochrane RCT data extraction.

Study_Characteristics and Intervention use Cochrane-specific fields.
Outcome models (Dichotomous, Continuous, Other) mirror the standard RCT models
and can be edited in place as Cochrane requirements evolve.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared helper models
# ---------------------------------------------------------------------------


class OutcomeTypes(BaseModel):
    value: str = Field(
        description="Outcome category. Choose one of: 'Adverse Event', 'Weight Outcome', 'Mental Health Outcome', 'Physical Activity Outcome', 'Other'"
    )


class OutcomeTimePoint(BaseModel):
    time_point_category: str = Field(
        description="Outcome time point. Choose one of: 'Baseline', 'Follow-up'"
    )
    time_point_detail: str = Field(
        description="The actual time point value as reported for this outcome"
    )


# ---------------------------------------------------------------------------
# Study characteristics (Cochrane-specific)
# ---------------------------------------------------------------------------


class Study_Characteristics(BaseModel):
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
# Outcomes (same as RCT for now — edit in place as Cochrane requirements evolve)
# ---------------------------------------------------------------------------


class Dichotomous_Outcome(BaseModel):
    outcome_name: str = Field(
        description="Name of the dichotomous (binary event) outcome"
    )
    outcome_definition: str = Field(
        description="Provide a definition of the outcome. For example, if it was participation in an activity, the events might relate to present/absent. Be as precise as possible to define what the events represent."
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    group_a_N: str = Field(
        description="Total number of participants analysed in group A for this outcome"
    )
    group_b_N: str = Field(
        description="Total number of participants analysed in group B for this outcome"
    )
    group_a_Events: str = Field(description="Number of events observed in group A")
    group_b_Events: str = Field(description="Number of events observed in group B")
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between the groups. Do not extrapolate. This information is often presented in tables."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? For example, were there assumptions made for intention to treat (ITT) analysis. Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by individuals, clusters, groups)? Note that a cluster RCT can have outcomes analysed at an individual level. Leave blank if not reported."
    )
    group_labels: dict[str, str] = Field(
        description=(
            "Maps 'group_a' and 'group_b' to the intervention group name each set of "
            "data corresponds to. Example: {'group_a': 'Intervention X', 'group_b': 'Placebo'}"
        )
    )
    supplementary_info: str = Field(
        description="If applicable, brief description of additional context. Flag up if any of the extracted numbers for this outcomes are percentages or otherwise not raw counts."
    )
    location_info: str = Field(
        description="If applicable, brief description of where in the source documents this outcome data was found (e.g. 'Table 2', section 'Results' or 'Figure 3'). This is to help with traceability and verification of the extracted data."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "outcome_type",
            "outcome_name",
            "outcome_definition",
            "group_a_label",
            "group_b_label",
            "group_a_N",
            "group_b_N",
            "group_a_Events",
            "group_b_Events",
            "imputation_of_missing_data",
            "power",
            "unit_of_analysis",
            "supplementary_info",
            "location_info",
            "baseline_imbalances",
            "outcome_category",
            "outcome_time_point",
            "outcome_time_point_detail",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "outcome_type": "dichotomous",
            "outcome_name": self.outcome_name,
            "outcome_definition": self.outcome_definition,
            "group_a_label": self.group_labels.get("group_a", ""),
            "group_b_label": self.group_labels.get("group_b", ""),
            "group_a_N": self.group_a_N,
            "group_b_N": self.group_b_N,
            "group_a_Events": self.group_a_Events,
            "group_b_Events": self.group_b_Events,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
            "baseline_imbalances": self.baseline_imbalances,
            "outcome_category": self.outcome_category.value,
            "outcome_time_point": self.outcome_time_point.time_point_category,
            "outcome_time_point_detail": self.outcome_time_point.time_point_detail,
        }


class Continuous_Outcome(BaseModel):
    outcome_name: str = Field(description="Name of the continuous outcome")
    outcome_definition: str = Field(
        description="Provide a definition of the outcome. For example, if it was a scale or any other measurement. Be as precise as possible. "
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    group_a_N: str = Field(
        description="Total number of participants analysed in group A for this outcome"
    )
    group_b_N: str = Field(
        description="Total number of participants analysed in group B for this outcome"
    )
    group_a_mean: str = Field(description="Mean value for group A")
    group_b_mean: str = Field(description="Mean value for group B")
    group_a_standard_deviation: str = Field(
        description="Standard deviation for group A"
    )
    group_b_standard_deviation: str = Field(
        description="Standard deviation for group B"
    )
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between the groups. Do not extrapolate. This information is often presented in tables."
    )
    unit_of_measurement: str = Field(
        description="What was the unit of measurement for this outcome? Leave blank if not reported."
    )
    scales_upper_and_lower_limits: str = Field(
        description="For this outcome, on the scale, tool, or method it was measured, indicate whether high or low score is good. Leave blank if not reported."
    )
    is_outcome_tool_validated: str = Field(
        description="Is the tool, method or scale used to measure this outcome validated, and what evidence is available? Leave blank if not reported."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? For example, were there assumptions made for intention to treat (ITT) analysis. Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    effect_estimates: str = Field(
        description="Extract results on confidence intervals, p-values, or any other relevant effect estimates related to this outcome if available. Leave blank if not reported."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by individuals, clusters, groups)? Note that a cluster RCT can have outcomes analysed at an individual level. Leave blank if not reported."
    )
    group_labels: dict[str, str] = Field(
        description=(
            "Maps 'group_a' and 'group_b' to the intervention group name each set of "
            "data corresponds to. Example: {'group_a': 'Drug X', 'group_b': 'Placebo'}"
        )
    )
    supplementary_info: str = Field(
        description="If applicable, brief description of additional context. Flag up if any of the extracted numbers for this outcomes are percentages or otherwise not raw counts."
    )
    location_info: str = Field(
        description="If applicable, brief description of where in the source documents this outcome data was found (e.g. 'Table 2', section 'Results' or 'Figure 3'). This is to help with traceability and verification of the extracted data."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "outcome_type",
            "outcome_name",
            "outcome_definition",
            "group_a_label",
            "group_b_label",
            "group_a_N",
            "group_b_N",
            "group_a_mean",
            "group_b_mean",
            "group_a_standard_deviation",
            "group_b_standard_deviation",
            "unit_of_measurement",
            "scales_upper_and_lower_limits",
            "is_outcome_tool_validated",
            "imputation_of_missing_data",
            "power",
            "effect_estimates",
            "unit_of_analysis",
            "supplementary_info",
            "location_info",
            "baseline_imbalances",
            "outcome_category",
            "outcome_time_point",
            "outcome_time_point_detail",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "outcome_type": "continuous",
            "outcome_name": self.outcome_name,
            "outcome_definition": self.outcome_definition,
            "group_a_label": self.group_labels.get("group_a", ""),
            "group_b_label": self.group_labels.get("group_b", ""),
            "group_a_N": self.group_a_N,
            "group_b_N": self.group_b_N,
            "group_a_mean": self.group_a_mean,
            "group_b_mean": self.group_b_mean,
            "group_a_standard_deviation": self.group_a_standard_deviation,
            "group_b_standard_deviation": self.group_b_standard_deviation,
            "unit_of_measurement": self.unit_of_measurement,
            "scales_upper_and_lower_limits": self.scales_upper_and_lower_limits,
            "is_outcome_tool_validated": self.is_outcome_tool_validated,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "effect_estimates": self.effect_estimates,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
            "baseline_imbalances": self.baseline_imbalances,
            "outcome_category": self.outcome_category.value,
            "outcome_time_point": self.outcome_time_point.time_point_category,
            "outcome_time_point_detail": self.outcome_time_point.time_point_detail,
        }


class Other_Outcome(BaseModel):
    outcome_name: str = Field(description="Name of the outcome")
    outcome_definition: str = Field(
        description="Provide a definition of the outcome, what was measured and what the measurement stands for. Be as precise as possible."
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    group_a_result: str = Field(description="Outcome results reported for group a")
    group_b_result: str = Field(description="Outcome results reported for group b")
    effect_estimates: str = Field(
        description="Extract results on confidence intervals, p-values, or any other relevanteffect estimates related to this outcome if available. Leave blank if not reported. "
    )
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between the groups. Do not extrapolate. This information is often presented in tables."
    )
    unit_of_measurement: str = Field(
        description="What was the unit of measurement for this outcome? Leave blank if not reported."
    )
    scales_upper_and_lower_limits: str = Field(
        description="For this outcome, on the smethod it was measured, indicate whether high or low score is good. Leave blank if not reported."
    )
    is_outcome_tool_validated: str = Field(
        description="Is the method used to measure this outcome validated, and what evidence is available? Leave blank if not reported."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? For example, were there assumptions made for intention to treat (ITT) analysis. Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by individuals, clusters, groups)? Note that a cluster RCT can have outcomes analysed at an individual level. Leave blank if not reported."
    )
    group_labels: dict[str, str] = Field(
        description=(
            "Maps 'group_a' and 'group_b' to the intervention group name each set of "
            "data corresponds to. Example: {'group_a': 'Drug X', 'group_b': 'Placebo'}"
        )
    )
    supplementary_info: str = Field(
        description="If applicable, brief description of additional context."
    )
    location_info: str = Field(
        description="If applicable, brief description of where in the source documents this outcome data was found (e.g. 'Table 2', section 'Results' or 'Figure 3'). This is to help with traceability and verification of the extracted data."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "outcome_type",
            "outcome_name",
            "outcome_definition",
            "unit_of_measurement",
            "scales_upper_and_lower_limits",
            "is_outcome_tool_validated",
            "imputation_of_missing_data",
            "power",
            "group_a_label",
            "group_b_label",
            "group_a_result",
            "group_b_result",
            "effect_estimates",
            "unit_of_analysis",
            "supplementary_info",
            "location_info",
            "baseline_imbalances",
            "outcome_category",
            "outcome_time_point",
            "outcome_time_point_detail",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "outcome_type": "other",
            "outcome_name": self.outcome_name,
            "outcome_definition": self.outcome_definition,
            "unit_of_measurement": self.unit_of_measurement,
            "scales_upper_and_lower_limits": self.scales_upper_and_lower_limits,
            "is_outcome_tool_validated": self.is_outcome_tool_validated,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "group_a_label": self.group_labels.get("group_a", ""),
            "group_b_label": self.group_labels.get("group_b", ""),
            "group_a_result": self.group_a_result,
            "group_b_result": self.group_b_result,
            "effect_estimates": self.effect_estimates,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
            "baseline_imbalances": self.baseline_imbalances,
            "outcome_category": self.outcome_category.value,
            "outcome_time_point": self.outcome_time_point.time_point_category,
            "outcome_time_point_detail": self.outcome_time_point.time_point_detail,
        }


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
    other_outcomes: list[Other_Outcome] = Field(
        default_factory=list,
        description="All other outcome types extracted from the study.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Study_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.study_characteristics.to_csv_row()

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        """Union of dichotomous, continuous and other fieldnames, in declaration order."""
        return list(
            dict.fromkeys(
                Dichotomous_Outcome.csv_fieldnames()
                + Continuous_Outcome.csv_fieldnames()
                + Other_Outcome.csv_fieldnames()
            )
        )
