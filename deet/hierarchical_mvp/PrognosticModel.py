"""Pydantic data models for prognostic study data extraction."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PrognosticStudy_Characteristics(BaseModel):
    study_author_contact_details: str = Field(
        description="Identify and extract the contact details for the corresponding author only. This is commonly an email address. If no specific corresponding author email is available, leave blank."
    )
    aim_of_study: str = Field(
        description="Be concise and use bullet points if there are several goals to the study. Only mention objectives stated as such by the authors; don't extrapolate. Provide the answer word for word from the relevant sentence."
    )
    design: str = Field(
        description="List all characteristics of the study design, such as whether it is a prospective or retrospective cohort study, registry-based study, or case-control study. If the study design is not mentioned, leave blank."
    )
    start_end_dates: str = Field(
        description="Identify and extract the start and end dates of the study. Ignore publication dates."
    )
    total_study_duration: str = Field(
        description="Extract the total duration of the study. If not explicitly provided, calculate from start and end dates and note the calculation."
    )
    follow_up_duration: str = Field(
        description="Extract the follow-up duration for participants. Include median and range if reported. Leave blank if not reported."
    )
    study_funding_sources: str = Field(
        description="Include all people or organizations that funded the study. Use bullet points for each funder. Include grant numbers if given. If no funding, answer 'no funding'. Leave blank if not reported."
    )
    possible_conflicts_of_interest: str = Field(
        description="Identify and extract any information on conflicts of interest reported by the authors. Leave blank if not reported."
    )
    population_description: str = Field(
        description="Identify and extract population characteristics of participants included in this study. Include the number enrolled if reported. Do not include inclusion or exclusion criteria here."
    )
    setting: str = Field(
        description="List the country or countries in which the study was conducted. Describe the specific location setting (e.g. hospital, registry, community). Be precise."
    )
    inclusion_criteria: str = Field(
        description="Identify and extract all inclusion criteria listed in the study. Report each criterion individually."
    )
    exclusion_criteria: str = Field(
        description="Identify and extract all exclusion criteria listed in the study. Report each criterion individually. Leave blank if not specified."
    )
    total_sample_size: str = Field(
        description="Extract the total number of participants included in the analysis. If available, also note the number of events (e.g. deaths, relapses)."
    )
    age: str = Field(
        description="Give the mean and standard deviation or median and range of participant ages. Do not provide age under inclusion or exclusion criteria."
    )
    sex: str = Field(
        description="Identify and extract the baseline characteristics of the sex or gender of participants. Leave blank if not mentioned."
    )
    race_ethnicity: str = Field(
        description="Identify and extract any reported race or ethnicity. If information is not in groups, provide an overall answer if available."
    )
    other_relevant_sociodemographics: str = Field(
        description="Identify and extract any reported sociodemographic characteristics (e.g. socioeconomic status). Leave blank if not reported."
    )
    statistical_analysis_method: str = Field(
        description="Describe the primary statistical methods used to assess prognostic factors (e.g. Cox proportional hazards regression, logistic regression, Kaplan-Meier). Include any adjustment variables mentioned."
    )
    study_registration_number: str = Field(
        description="Identify and extract any study registration number and the citation of any published protocol. Leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study_author_contact_details",
            "aim_of_study",
            "design",
            "start_end_dates",
            "total_study_duration",
            "follow_up_duration",
            "study_funding_sources",
            "possible_conflicts_of_interest",
            "population_description",
            "setting",
            "inclusion_criteria",
            "exclusion_criteria",
            "total_sample_size",
            "age",
            "sex",
            "race_ethnicity",
            "other_relevant_sociodemographics",
            "statistical_analysis_method",
            "study_registration_number",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "study_author_contact_details": self.study_author_contact_details,
            "aim_of_study": self.aim_of_study,
            "design": self.design,
            "start_end_dates": self.start_end_dates,
            "total_study_duration": self.total_study_duration,
            "follow_up_duration": self.follow_up_duration,
            "study_funding_sources": self.study_funding_sources,
            "possible_conflicts_of_interest": self.possible_conflicts_of_interest,
            "population_description": self.population_description,
            "setting": self.setting,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "total_sample_size": self.total_sample_size,
            "age": self.age,
            "sex": self.sex,
            "race_ethnicity": self.race_ethnicity,
            "other_relevant_sociodemographics": self.other_relevant_sociodemographics,
            "statistical_analysis_method": self.statistical_analysis_method,
            "study_registration_number": self.study_registration_number,
        }


class PrognosticFactor(BaseModel):
    factor_name: str = Field(
        description="The name of the prognostic factor as reported in the study (e.g. 'age > 65 years', 'stage III/IV', 'ECOG performance status ≥ 2')."
    )
    description: str = Field(
        description="Describe the prognostic factor in detail. Include how it was defined or categorised, any cut-off values used, and its clinical relevance."
    )
    measurement_method: str = Field(
        description="Describe how the prognostic factor was measured or assessed (e.g. blood test, clinical assessment, imaging). Leave blank if not reported."
    )
    factor_type: str = Field(
        description="Classify the type of variable. Choose one of: 'binary', 'continuous', 'ordinal', 'categorical'. Use 'binary' if it was dichotomised."
    )
    reference_category: str = Field(
        description="Identify the reference category or group used in statistical comparisons (e.g. 'age ≤ 65 years', 'stage I/II'). Leave blank if not reported or not applicable."
    )
    group_n: str = Field(
        description="Number of participants in this prognostic factor group. Leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "factor_name",
            "description",
            "measurement_method",
            "factor_type",
            "reference_category",
            "group_n",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "factor_name": self.factor_name,
            "description": self.description,
            "measurement_method": self.measurement_method,
            "factor_type": self.factor_type,
            "reference_category": self.reference_category,
            "group_n": self.group_n,
        }


class HazardRatioOutcome(BaseModel):
    outcome_name: str = Field(
        description="Name of the outcome for which the hazard ratio is reported (e.g. 'overall survival', 'progression-free survival', 'cancer-specific mortality')."
    )
    outcome_category: str = Field(
        description="Category of the outcome. Choose one of: 'Survival', 'Recurrence/Relapse', 'Functional', 'Quality of Life', 'Biomarker', 'Other'."
    )
    outcome_time_point: str = Field(
        description="The time point at which the outcome was assessed (e.g. '5-year', '10-year', 'end of follow-up'). Leave blank if not specified."
    )
    group_n: str = Field(
        description="Number of participants in the group or subgroup for which this hazard ratio is reported. Leave blank if not reported."
    )
    group_hazard_ratio: str = Field(
        description="The hazard ratio value as reported in the study. Report exactly as written (e.g. '1.45'). Use 'NR' if not reported."
    )
    group_confidence_interval: str = Field(
        description="The confidence interval for the hazard ratio (e.g. '95% CI: 1.12–1.88'). Use 'NR' if not reported."
    )
    group_p_value: str = Field(
        description="The p-value associated with the hazard ratio. Report exactly as written. Use 'NR' if not reported."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding power calculations and sample size. Be as detailed as possible. Leave blank if not reported."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome (e.g. individuals, events)? Leave blank if not reported."
    )
    group_label: str = Field(
        description="The label of the group or subgroup for which this hazard ratio is reported (e.g. 'age > 65 years', 'high-risk', 'stage III/IV'). This should correspond to one of the identified PrognosticFactor names."
    )
    supplementary_info: str = Field(
        description="Brief description of additional context, adjustments, or caveats for this outcome. Note if the hazard ratio is adjusted or unadjusted and what variables were adjusted for."
    )
    location_info: str = Field(
        description="Brief description of where in the source document this outcome data was found (e.g. 'Table 2', 'Figure 3', section 'Results'). This aids traceability and verification."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "outcome_type",
            "outcome_name",
            "outcome_category",
            "outcome_time_point",
            "group_label",
            "group_n",
            "group_hazard_ratio",
            "group_confidence_interval",
            "group_p_value",
            "imputation_of_missing_data",
            "power",
            "unit_of_analysis",
            "supplementary_info",
            "location_info",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "outcome_type": "hazard_ratio",
            "outcome_name": self.outcome_name,
            "outcome_category": self.outcome_category,
            "outcome_time_point": self.outcome_time_point,
            "group_label": self.group_label,
            "group_n": self.group_n,
            "group_hazard_ratio": self.group_hazard_ratio,
            "group_confidence_interval": self.group_confidence_interval,
            "group_p_value": self.group_p_value,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
        }


class OtherPrognosticOutcome(BaseModel):
    outcome_name: str = Field(
        description="Name of the outcome (e.g. 'tumour response rate', 'quality of life score', 'complication rate')."
    )
    outcome_category: str = Field(
        description="Category of the outcome. Choose one of: 'Survival', 'Recurrence/Relapse', 'Functional', 'Quality of Life', 'Biomarker', 'Other'."
    )
    outcome_time_point: str = Field(
        description="The time point at which the outcome was assessed. Leave blank if not specified."
    )
    group_n: str = Field(
        description="Number of participants in the group for which this outcome is reported. Leave blank if not reported."
    )
    effect_estimate_average: str = Field(
        description="Mean or average value of the outcome for this group as reported. Use 'NR' if not reported."
    )
    effect_estimate_median: str = Field(
        description="Median value of the outcome for this group as reported. Use 'NR' if not reported."
    )
    effect_estimate_sd: str = Field(
        description="Standard deviation of the outcome for this group as reported. Use 'NR' if not reported."
    )
    other_effect_value: str = Field(
        description="Any other reported effect estimate not captured above (e.g. odds ratio, rate, percentage, interquartile range). Report exactly as written. Use 'NR' if not reported."
    )
    unit_of_measurement: str = Field(
        description="The unit of measurement for this outcome (e.g. months, score points, %). Leave blank if not reported."
    )
    high_is_good: str = Field(
        description="Indicate whether a higher value for this outcome is desirable ('yes'), undesirable ('no'), or context-dependent ('unclear'). Leave blank if not applicable."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding power calculations and sample size. Be as detailed as possible. Leave blank if not reported."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome (e.g. individuals, events)? Leave blank if not reported."
    )
    group_label: str = Field(
        description="The label of the group or subgroup for which this outcome is reported. Should correspond to one of the identified PrognosticFactor names."
    )
    supplementary_info: str = Field(
        description="Brief description of additional context, adjustments, or caveats for this outcome."
    )
    location_info: str = Field(
        description="Brief description of where in the source document this outcome data was found (e.g. 'Table 2', 'Figure 3', section 'Results'). This aids traceability and verification."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "outcome_type",
            "outcome_name",
            "outcome_category",
            "outcome_time_point",
            "group_label",
            "group_n",
            "effect_estimate_average",
            "effect_estimate_median",
            "effect_estimate_sd",
            "other_effect_value",
            "unit_of_measurement",
            "high_is_good",
            "imputation_of_missing_data",
            "power",
            "unit_of_analysis",
            "supplementary_info",
            "location_info",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "outcome_type": "other_prognostic",
            "outcome_name": self.outcome_name,
            "outcome_category": self.outcome_category,
            "outcome_time_point": self.outcome_time_point,
            "group_label": self.group_label,
            "group_n": self.group_n,
            "effect_estimate_average": self.effect_estimate_average,
            "effect_estimate_median": self.effect_estimate_median,
            "effect_estimate_sd": self.effect_estimate_sd,
            "other_effect_value": self.other_effect_value,
            "unit_of_measurement": self.unit_of_measurement,
            "high_is_good": self.high_is_good,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
        }


class PrognosticStudy(BaseModel):
    study_characteristics: PrognosticStudy_Characteristics = Field(
        description="Study-level metadata including design, population, setting, and statistical methods."
    )
    prognostic_factors: list[PrognosticFactor] = Field(
        description="All prognostic factors (predictors) identified in the study."
    )
    hazard_ratio_outcomes: list[HazardRatioOutcome] = Field(
        default_factory=list,
        description="All outcomes reported as hazard ratios.",
    )
    other_prognostic_outcomes: list[OtherPrognosticOutcome] = Field(
        default_factory=list,
        description="All other prognostic outcomes not reported as hazard ratios.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return PrognosticStudy_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.study_characteristics.to_csv_row()

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        """Union of hazard ratio and other outcome fieldnames, in declaration order."""
        return list(
            dict.fromkeys(
                HazardRatioOutcome.csv_fieldnames()
                + OtherPrognosticOutcome.csv_fieldnames()
            )
        )
