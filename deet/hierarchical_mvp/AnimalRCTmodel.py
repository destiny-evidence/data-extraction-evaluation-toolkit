"""Pydantic data models for Animal RCT outcome data extraction.

Unlike RCTmodel.py/ObesityRCTmodel.py, this study type has two distinct
intervention lists (InductionIntervention, AssessmentIntervention) instead of
a single Intervention list, and outcomes are reported for a unique
combination of induction and assessment intervention rather than for a
group_a/group_b comparison.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutcomeTypes(BaseModel):
    value: str = Field(
        description="Outcome category. Choose one of: 'Adverse Event', 'Weight Outcome', 'Mental Health Outcome', 'Physical Activity Outcome', 'Survival', 'Disease Progression', 'Other'"
    )


class OutcomeTimePoint(BaseModel):
    time_point_category: str = Field(
        description="Outcome time point. Choose one of: 'Baseline', 'Post-intervention', 'Follow-up'"
    )
    time_point_detail: str = Field(
        description="The actual time point value as reported for this outcome"
    )


class InductionIntervention(BaseModel):
    intervention_to_induce_name: str = Field(
        description="Identify and extract the name of the intervention used to induce the condition, injury, disease model, or subpopulation in this study arm. This could be in the form of abbreviations."
    )
    intervention_to_induce_description: str = Field(
        description="Describe this induction intervention. Note the frequency, duration, and amount or dose used to induce the condition or model. Be as precise as possible and report as written in the text."
    )
    n_in_subpopulation: str = Field(
        description="Extract the total number of animals or units allocated to this subpopulation following the induction intervention. Leave blank if not reported."
    )
    n_lost_or_missing: str = Field(
        description="Identify and extract the number of animals or units lost, excluded, or missing from this subpopulation, along with the reason if provided. Don't extrapolate. Leave blank if not reported."
    )
    type_of_subpopulation: str = Field(
        description="Classify the subpopulation created by this induction intervention. Choose one of: 'naive', 'sham', 'model', 'other', 'not reported'."
    )
    type_of_model_induction: str = Field(
        description="Classify the method used to induce the model or condition. Choose one of: 'surgery', 'injury', 'drug/toxicity', 'gene modification', 'other', 'not reported'."
    )
    resource_requirements: str = Field(
        description="Identify and extract information on the resources required for this induction intervention. This may include the number of staff, the equipment required and the funds required. Do not extrapolate and report as written in the text. Leave blank if not reported."
    )
    dosage: str = Field(
        description="Identify and extract the dosage or amount used as part of this induction intervention, if applicable. Leave blank if not reported."
    )
    is_control: bool = Field(
        description="Whether this subpopulation represents the control or comparator group for the induction step."
    )
    sex_in_subpopulation: str = Field(
        description="Identify and extract the sex distribution of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    strain_in_subpopulation: str = Field(
        description="Identify and extract the strain of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    age_in_subpopulation: str = Field(
        description="Identify and extract the age of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    weight_in_subpopulation: str = Field(
        description="Identify and extract the body weight of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "intervention_to_induce_name",
            "intervention_to_induce_description",
            "n_in_subpopulation",
            "n_lost_or_missing",
            "type_of_subpopulation",
            "type_of_model_induction",
            "resource_requirements",
            "dosage",
            "is_control",
            "sex_in_subpopulation",
            "strain_in_subpopulation",
            "age_in_subpopulation",
            "weight_in_subpopulation",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "intervention_to_induce_name": self.intervention_to_induce_name,
            "intervention_to_induce_description": self.intervention_to_induce_description,
            "n_in_subpopulation": self.n_in_subpopulation,
            "n_lost_or_missing": self.n_lost_or_missing,
            "type_of_subpopulation": self.type_of_subpopulation,
            "type_of_model_induction": self.type_of_model_induction,
            "resource_requirements": self.resource_requirements,
            "dosage": self.dosage,
            "is_control": self.is_control,
            "sex_in_subpopulation": self.sex_in_subpopulation,
            "strain_in_subpopulation": self.strain_in_subpopulation,
            "age_in_subpopulation": self.age_in_subpopulation,
            "weight_in_subpopulation": self.weight_in_subpopulation,
        }


class AssessmentIntervention(BaseModel):
    intervention_to_assess_name: str = Field(
        description="Identify and extract the name of the intervention being assessed (e.g. treatment, drug, procedure) in this study arm. This could be in the form of abbreviations."
    )
    intervention_to_assess_description: str = Field(
        description="Describe this assessment intervention. Note the frequency, duration, and amount or dose of the intervention. Be as precise as possible: if duration, frequency or dose are mentioned, they need to be in the answer."
    )
    n_in_subpopulation: str = Field(
        description="Extract the total number of animals or units allocated to this subpopulation for this assessment intervention."
    )
    n_lost_or_missing: str = Field(
        description="Identify and extract the number of animals or units lost, excluded, or missing from this subpopulation, along with the reason if provided. Don't extrapolate. Leave blank if not mentioned."
    )
    delivery: str = Field(
        description="Identify and extract the delivery method of the assessment intervention. This may include the medium, route of administration, intensity and fidelity. Report as written in the text."
    )
    dosage: str = Field(
        description="Identify and extract the dosage or amount used as part of this assessment intervention, if applicable. Leave blank if not reported."
    )
    resource_requirements: str = Field(
        description="Identify and extract information on the resources required for this assessment intervention. This may include the number of staff, the equipment required and the funds required. Do not extrapolate and report as written in the text. Leave blank if not reported."
    )
    is_control: bool = Field(
        description="Whether this subpopulation represents the control or comparator group for the assessment step."
    )
    sex_in_subpopulation: str = Field(
        description="Identify and extract the sex distribution of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    strain_in_subpopulation: str = Field(
        description="Identify and extract the strain of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    age_in_subpopulation: str = Field(
        description="Identify and extract the age of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )
    weight_in_subpopulation: str = Field(
        description="Identify and extract the body weight of the animals within this subpopulation, if reported separately from the overall study population. Leave blank if not reported."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "intervention_to_assess_name",
            "intervention_to_assess_description",
            "n_in_subpopulation",
            "n_lost_or_missing",
            "delivery",
            "dosage",
            "resource_requirements",
            "is_control",
            "sex_in_subpopulation",
            "strain_in_subpopulation",
            "age_in_subpopulation",
            "weight_in_subpopulation",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "intervention_to_assess_name": self.intervention_to_assess_name,
            "intervention_to_assess_description": self.intervention_to_assess_description,
            "n_in_subpopulation": self.n_in_subpopulation,
            "n_lost_or_missing": self.n_lost_or_missing,
            "delivery": self.delivery,
            "dosage": self.dosage,
            "resource_requirements": self.resource_requirements,
            "is_control": self.is_control,
            "sex_in_subpopulation": self.sex_in_subpopulation,
            "strain_in_subpopulation": self.strain_in_subpopulation,
            "age_in_subpopulation": self.age_in_subpopulation,
            "weight_in_subpopulation": self.weight_in_subpopulation,
        }


class Dichotomous_Outcome(BaseModel):
    outcome_name: str = Field(
        description="Name of the dichotomous (binary event) outcome"
    )
    outcome_definition: str = Field(
        description="Provide a definition of the outcome and how it was defined. For example, if it was an adverse event. Be as precise as possible."
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    intervention_to_induce_name: str = Field(
        description="The name of the induction intervention (subpopulation) that this outcome data corresponds to."
    )
    intervention_to_assess_name: str = Field(
        description="The name of the assessment intervention that this outcome data corresponds to."
    )
    N: str = Field(
        description="Total number of animals or units analysed for this outcome, for this combination of induction and assessment intervention."
    )
    Events: str = Field(
        description="Number of events observed for this outcome, for this combination of induction and assessment intervention."
    )
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between this group and others. Do not extrapolate. This information is often presented in tables."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    is_outcome_tool_validated: str = Field(
        description="Is the tool or method used to measure this outcome validated, and what evidence is available? Leave blank if not reported."
    )
    measurement_reporting: str = Field(
        description="Extract who measured or reported this outcome. Provide any further detail that is relevant to the person measuring or reporting such as standardisations, training, intra and inter-observer reliability. Leave blank if not reported."
    )
    statistical_methods_used: str = Field(
        description="Extract all information on the statistical methods for measuring this outcome. Do not extrapolate. If applicable, the answer should consist of general mathematical techniques, tools or tests, such as regression, t-test, ANOVA, chi-square test, etc. Leave blank if no statistical techniques were used."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by animal, litter, cage, group. Leave blank if not reported."
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
            "intervention_to_induce_name",
            "intervention_to_assess_name",
            "N",
            "Events",
            "imputation_of_missing_data",
            "power",
            "is_outcome_tool_validated",
            "measurement_reporting",
            "statistical_methods_used",
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
            "intervention_to_induce_name": self.intervention_to_induce_name,
            "intervention_to_assess_name": self.intervention_to_assess_name,
            "N": self.N,
            "Events": self.Events,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "is_outcome_tool_validated": self.is_outcome_tool_validated,
            "measurement_reporting": self.measurement_reporting,
            "statistical_methods_used": self.statistical_methods_used,
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
        description="Provide a definition of the outcome and how it was defined. Be as precise as possible."
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    intervention_to_induce_name: str = Field(
        description="The name of the induction intervention (subpopulation) that this outcome data corresponds to."
    )
    intervention_to_assess_name: str = Field(
        description="The name of the assessment intervention that this outcome data corresponds to."
    )
    N: str = Field(
        description="Total number of animals or units analysed for this outcome, for this combination of induction and assessment intervention."
    )
    mean: str = Field(
        description="Mean value for this outcome, for this combination of induction and assessment intervention."
    )
    standard_deviation: str = Field(
        description="Standard deviation for this outcome, for this combination of induction and assessment intervention."
    )
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between this group and others. Do not extrapolate. This information is often presented in tables."
    )
    unit_of_measurement: str = Field(
        description="Unit of measurement for this outcome. Leave blank if not reported."
    )
    scales_upper_and_lower_limits: str = Field(
        description="For this outcome, on the scale, tool, or method it was measured, indicate whether high or low score is good. Leave blank if not reported."
    )
    is_outcome_tool_validated: str = Field(
        description="Is the tool, method or scale used to measure this outcome validated, and what evidence is available? Leave blank if not reported."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    effect_estimates: str = Field(
        description="Extract results on other effect estimates such as confidence intervals, p-values, or any other relevant effect estimates related to this outcome if available. Leave blank if not reported."
    )
    minimally_important_difference: str = Field(
        description="What is the minimally important difference that is considered a meaningful change for this outcome? Leave blank if not reported."
    )
    measurement_reporting: str = Field(
        description="Extract who measured or reported this outcome. Provide any further detail that is relevant to the person measuring or reporting such as standardisations, training, intra and inter-observer reliability. Leave blank if not reported."
    )
    statistical_methods_used: str = Field(
        description="Extract all information on the statistical methods for measuring this outcome. Do not extrapolate. If applicable, the answer should consist of general mathematical techniques, tools or tests, such as regression, t-test, ANOVA, chi-square test, etc. Leave blank if no statistical techniques were used."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by animal, litter, cage, group. Leave blank if not reported."
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
            "intervention_to_induce_name",
            "intervention_to_assess_name",
            "N",
            "mean",
            "standard_deviation",
            "unit_of_measurement",
            "scales_upper_and_lower_limits",
            "is_outcome_tool_validated",
            "imputation_of_missing_data",
            "power",
            "effect_estimates",
            "minimally_important_difference",
            "measurement_reporting",
            "statistical_methods_used",
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
            "intervention_to_induce_name": self.intervention_to_induce_name,
            "intervention_to_assess_name": self.intervention_to_assess_name,
            "N": self.N,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "unit_of_measurement": self.unit_of_measurement,
            "scales_upper_and_lower_limits": self.scales_upper_and_lower_limits,
            "is_outcome_tool_validated": self.is_outcome_tool_validated,
            "imputation_of_missing_data": self.imputation_of_missing_data,
            "power": self.power,
            "effect_estimates": self.effect_estimates,
            "minimally_important_difference": self.minimally_important_difference,
            "measurement_reporting": self.measurement_reporting,
            "statistical_methods_used": self.statistical_methods_used,
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
        description="Provide a definition of the outcome and how it was defined and operationalised. Be as precise as possible."
    )
    outcome_category: OutcomeTypes = Field(
        description="Assign the best fitting outcome category defined in this classification scheme."
    )
    intervention_to_induce_name: str = Field(
        description="The name of the induction intervention (subpopulation) that this outcome data corresponds to."
    )
    intervention_to_assess_name: str = Field(
        description="The name of the assessment intervention that this outcome data corresponds to."
    )
    outcome_time_point: OutcomeTimePoint = Field(
        description="Assign the best fitting outcome time point defined in this classification scheme."
    )
    result: str = Field(
        description="Outcome results reported for this combination of induction and assessment intervention."
    )
    effect_estimates: str = Field(
        description="Extract results on confidence intervals, p-values, or any other relevant effect estimates related to this outcome if available. Leave blank if not reported."
    )
    baseline_imbalances: str = Field(
        description="Identify if there were any baseline differences between this group and others. Do not extrapolate. This information is often presented in tables."
    )
    unit_of_measurement: str = Field(
        description="Unit of measurement for this outcome. Leave blank if not reported."
    )
    scales_upper_and_lower_limits: str = Field(
        description="For this outcome, on the scale, tool, or method it was measured, indicate whether high or low score is good. Leave blank if not reported."
    )
    is_outcome_tool_validated: str = Field(
        description="Is the method used to measure this outcome validated, and what evidence is available? Leave blank if not reported."
    )
    imputation_of_missing_data: str = Field(
        description="Was there imputation of missing data used for the analysis of this outcome? Be as detailed as possible. Leave blank if not reported."
    )
    power: str = Field(
        description="Extract all text regarding the power, sample size calculations and level of power achieved? Be as detailed as possible. Leave blank if not reported."
    )
    minimally_important_difference: str = Field(
        description="What is the minimally important difference that is considered a meaningful change for this outcome? Leave blank if not reported."
    )
    measurement_reporting: str = Field(
        description="Extract who measured or reported this outcome. Provide any further detail that is relevant to the person measuring or reporting such as standardisations, training, intra and inter-observer reliability. Leave blank if not reported."
    )
    statistical_methods_used: str = Field(
        description="Extract all information on the statistical methods for measuring this outcome. Do not extrapolate. If applicable, the answer should consist of general mathematical techniques, tools or tests, such as regression, t-test, ANOVA, chi-square test, etc. Leave blank if no statistical techniques were used."
    )
    unit_of_analysis: str = Field(
        description="What was the unit of analysis for this outcome? For example, by animal, litter, cage, group. Leave blank if not reported."
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
            "minimally_important_difference",
            "measurement_reporting",
            "statistical_methods_used",
            "intervention_to_induce_name",
            "intervention_to_assess_name",
            "result",
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
            "minimally_important_difference": self.minimally_important_difference,
            "measurement_reporting": self.measurement_reporting,
            "statistical_methods_used": self.statistical_methods_used,
            "intervention_to_induce_name": self.intervention_to_induce_name,
            "intervention_to_assess_name": self.intervention_to_assess_name,
            "result": self.result,
            "effect_estimates": self.effect_estimates,
            "unit_of_analysis": self.unit_of_analysis,
            "supplementary_info": self.supplementary_info,
            "location_info": self.location_info,
            "baseline_imbalances": self.baseline_imbalances,
            "outcome_category": self.outcome_category.value,
            "outcome_time_point": self.outcome_time_point.time_point_category,
            "outcome_time_point_detail": self.outcome_time_point.time_point_detail,
        }


class Study_Characteristics(BaseModel):
    study_author_contact_details: str = Field(
        description="Identify and extract the contact details for the corresponding author only. This is commonly an email address. If multiple email addresses are provided, select only the one identified as the corresponding author. If no specific corresponding author email is available, leave the field blank."
    )
    aim_of_study: str = Field(
        description="Be concise and use bullet points if there are several goals to the study. This could be reported as study aim also. Only mention objectives that are stated as such by the authors; don't extrapolate. Provide the answer word for word from the relevant sentence."
    )
    design: str = Field(
        description="List all characteristics of study design, such as whether it was a randomised controlled animal trial. Extract brief descriptions on each of the study arms and experiments. If the study design is not mentioned, leave the answer blank."
    )
    total_study_duration: str = Field(
        description="Extract the total duration of the study. If the duration is not explicitly provided, locate the start and end dates, calculate the duration by subtracting the start date from the end date, and note the calculation."
    )
    setting: str = Field(
        description="Identify and extract the setting in which the study was conducted, e.g. laboratory, animal facility, institution, country. Be as precise as possible. Leave the answer blank if not specified."
    )
    inclusion_criteria: str = Field(
        description="Identify and extract all inclusion criteria listed in the study for the animals or units used. Report each criterion individually."
    )
    exclusion_criteria: str = Field(
        description="Identify and extract all exclusion criteria listed in the study for the animals or units used. Don't extrapolate. Report answers as closely to the study text as possible. Leave the answer blank if not specified."
    )
    total_no_randomised: str = Field(
        description="If randomization was done, give the number of animals or units randomized at the baseline or first phase of the study, not necessarily the number who were actually treated or analyzed. If available, state the number per each trial arm together with the arm names, using bullet points, and ensure that they add up to the total."
    )
    species: str = Field(
        description="Identify and extract the species of animal used in this study, e.g. mouse, rat, rabbit, pig, non-human primate. Leave blank if not reported."
    )
    sex: str = Field(
        description="Identify and extract the baseline characteristics of the sex of the animals used. If multiple groups are available, provide the answer for all groups in bullet points. This answer is often presented in tables."
    )
    strain: str = Field(
        description="Identify and extract the strain or genetic background of the animals used in this study, e.g. C57BL/6, Sprague-Dawley. Leave blank if not reported."
    )
    age: str = Field(
        description="Give the mean and standard deviation (SD) of the age of the animals at the start of the study; provide either a mean and SD across all animals, or a mean and SD by group, or both. You may provide an age range if a mean is not available. Leave blank if not reported."
    )
    weight: str = Field(
        description="Give the mean and standard deviation (SD) of the body weight of the animals at the start of the study; provide either a mean and SD across all animals, or a mean and SD by group, or both. You may provide a weight range if a mean is not available. Leave blank if not reported."
    )
    origin: str = Field(
        description="Identify and extract the origin of the animals used in the study. Choose one or both of: 'bred-in-house', 'supplier'. If a supplier is named, include the name. Leave blank if not reported."
    )
    no_missing_reasons: str = Field(
        description="Identify and extract the number of missing animals or units, defined as those that died, withdrew, or were excluded between allocation and follow-up. Extract the reasons for missingness if provided. If missing animals are reported by group, list each group separately and use bullet points. Do not extract any information about imputation of missing data. Do not extrapolate; leave blank if no missing animals are reported."
    )
    study_funding_sources: str = Field(
        description="Include all people or organizations that funded the study. Answer using bullet points with a separate point for each funder. Include as much detail as possible about who provided how much funding, and who funded what. Include grant numbers if they are given in the paper. If a paper explicitly says there was no funding, answer 'no funding'. If there is no funding information, leave blank."
    )
    possible_conflicts_of_interest: str = Field(
        description="Identify and extract any information on conflicts of interest reported by the authors. If there is no information available on conflicts of interest, leave the field blank."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study_author_contact_details",
            "aim_of_study",
            "design",
            "total_study_duration",
            "setting",
            "inclusion_criteria",
            "exclusion_criteria",
            "total_no_randomised",
            "species",
            "sex",
            "strain",
            "age",
            "weight",
            "origin",
            "no_missing_reasons",
            "study_funding_sources",
            "possible_conflicts_of_interest",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "study_author_contact_details": self.study_author_contact_details,
            "aim_of_study": self.aim_of_study,
            "design": self.design,
            "total_study_duration": self.total_study_duration,
            "setting": self.setting,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "total_no_randomised": self.total_no_randomised,
            "species": self.species,
            "sex": self.sex,
            "strain": self.strain,
            "age": self.age,
            "weight": self.weight,
            "origin": self.origin,
            "no_missing_reasons": self.no_missing_reasons,
            "study_funding_sources": self.study_funding_sources,
            "possible_conflicts_of_interest": self.possible_conflicts_of_interest,
        }


class Study(BaseModel):
    study_characteristics: Study_Characteristics = Field(
        description="Study-level metadata including design, population, setting, and key conclusions."
    )
    induction_interventions: list[InductionIntervention] = Field(
        description="All induction interventions (subpopulations/arms created to induce a condition or model) identified in the trial."
    )
    assessment_interventions: list[AssessmentIntervention] = Field(
        description="All assessment interventions (treatments/procedures being assessed) identified in the trial."
    )
    dichotomous_outcomes: list[Dichotomous_Outcome] = Field(
        default_factory=list,
        description="All dichotomous (binary event) outcomes extracted from the study, reporting numbers of events.",
    )
    continuous_outcomes: list[Continuous_Outcome] = Field(
        default_factory=list,
        description="All continuous outcomes extracted from the study, reporting means and standard deviations.",
    )
    other_outcomes: list[Other_Outcome] = Field(
        default_factory=list,
        description="All other outcome types and result types (numeric/qualitative/descriptive) outcomes extracted from the study, reporting results that are not related to events or means/standard deviations.",
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
