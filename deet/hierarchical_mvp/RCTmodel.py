from __future__ import annotations

from pydantic import BaseModel, Field


class OutcomeTypes(
    BaseModel
):  # I think this can be structured better for sure, in terms of data validation, in-and export etc.
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


class InterventionType(BaseModel):
    type_of_intervention: str = Field(
        description="The type of this study arm in the context of this study. Choose one of: 'Intervention', 'Control'"
    )


class Intervention(BaseModel):
    group_name: str = Field(
        description="Identify and extract the name of the intervention group or groups from this study. This could be in the form of abbreviations. For example, 'mindfulness-based group', 'ASS high dose'. "
    )
    intervention_type: InterventionType = Field()
    description: str = Field(
        description="Describe this intervention arm. If only some participants received the treatment in this arm, note that. Note the frequency, duration, and amount or dose of the intervention. Be as precise as possible: if duration, frequency or dose are mentioned, they need to be in the answer. If there are multiple components to the intervention (e.g. physical activity plus diet), state all of them."
    )
    no_randomised_per_group: str = Field(
        description="Extract what was the number of individuals or other units randomised to this group at the commencement of the study. Specify whether the unit is individuals or clusters."
    )
    no_missing_and_reasons: str = Field(
        description="Identify and extract the number of missing participants or clusters within this arm of the trial, defined as those who withdrew from the study or were excluded between randomization and follow-up. Extract the reasons for missingness if provided (e.g., withdrawal, exclusion after randomization). Extract any details on whether missing participants were analysed. Do not extract any information about imputation of missing data. Do not extrapolate; leave blank if no missing participants are reported."
    )
    duration_of_intervention: str = Field(
        description="Give the time period between the beginning and end of the intervention. The duration includes the entire intervention period, but not recruitment or follow-up periods after the intervention ended. There may be multiple durations if participants received intervention of varying lengths. Be as precise as possible (i.e. '50 hours' is better than '2 days'). Don't extrapolate and report as written in text (e.g.  6 2-hour, biweekly sessions)."
    )
    timing: str = Field(
        description="Identify and extract details about the intervention timing. This may include the frequency and duration of each activity or session. Be as precise as possible and report as written in the text."
    )
    delivery: str = Field(
        description="Identify and extract the delivery method of the intervention. This may include the medium, intensity and fidelity. (e.g. Out-of-school program for youth and main adult food preparer including six sessions, monthly newsletters and booster sessions). Report as written in the text."
    )
    providers: str = Field(
        description="Identify and extract the providers of the intervention (i.e. the person delivering the intervention). If available, provide details about the number of providers, their profession, if they received training or supervision, if they are of a specific ethnicity or sex or gender. Only extract details from the intervention group and not the control group."
    )
    economic_information: str = Field(
        description="Identify and extract economic information of the intervention. This may include the cost of any equipment provided to the participants or the cost of training providers. Do not extrapolate and report as written in the text. Leave blank if not reported."
    )
    resource_requirements: str = Field(
        description="Identify and extract information of the resources required for the intervention. This may include the number of staff, the equipment required and the funds required. Do not extrapolate and report as written in the text. Leave blank if not reported."
    )
    compliance: str = Field(
        description="Identify and extraction information regarding the compliance of participants to the intervention. Key words that often describe this are 'compliance', 'adherence' and 'fidelity'. Note compliance here refers to participants adhering to the intervention."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "group_name",
            "no_randomised_per_group",
            "no_missing_and_reasons",
            "description",
            "duration_of_intervention",
            "timing",
            "delivery",
            "providers",
            "economic_information",
            "resource_requirements",
            "compliance",
            "intervention_type",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "group_name": self.group_name,
            "no_randomised_per_group": self.no_randomised_per_group,
            "no_missing_and_reasons": self.no_missing_and_reasons,
            "description": self.description,
            "duration_of_intervention": self.duration_of_intervention,
            "timing": self.timing,
            "delivery": self.delivery,
            "providers": self.providers,
            "economic_information": self.economic_information,
            "resource_requirements": self.resource_requirements,
            "compliance": self.compliance,
            "intervention_type": self.intervention_type.type_of_intervention,
        }


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


class Study_Characteristics(BaseModel):
    study_author_contact_details: str = Field(
        description="Identify and extract the contact details for the corresponding author only. This is commonly an email address. If multiple email addresses are provided, select only the one identified as the corresponding author. If no specific corresponding author email is available, leave the field blank."
    )
    aim_of_study: str = Field(
        description="Be concise and use bullet points if there are several goals to the study. This could be reported as study aim also.  Only mention objectives that are stated as such by the authors; don't extrapolate. Provide the answer word for word from the relevant sentence."
    )
    design: str = Field(
        description="List all characteristics of study design, such as whether it was randomised controlled trial or a cluster randomised controlled trial. Extract brief descriptions on each of the study arms. If the study design is not mentioned, leave the answer blank"
    )
    unit_of_allocation: str = Field(
        description="Identify and extract information about the unit of allocation that was used as part of the randomisation process. Common units that maybe be randomly allocated include 'schools', 'classes', or 'students' "
    )
    start_end_dates: str = Field(
        description="Identify and extract the start and end dates of the study. Ignore publication dates. If multiple dates are available, prioritize those labeled with keywords such as 'study start,' trial began,' or 'recruitment started' 'follow-up'."
    )
    total_study_duration: str = Field(
        description="Extract the total duration of the study. If the duration is not explicitly provided, locate the start and end dates, calculate the duration by subtracting the start date from the end date, and note the calculation."
    )
    study_funding_sources: str = Field(
        description="Include all people or organizations that funded the study. Answer using bullet points with a separate point for each funder. Include as much detail as possible about who provided how much funding, and who funded what. Include grant numbers if they are given in the paper. If a paper explicitly says there was no funding, answer 'no funding'. If there is no funding information, leave blank."
    )
    possible_conflicts_of_interest: str = Field(
        description="Identify and extract any information on conflicts of interest reported by the authors. If there is no information available on conflicts of interest, leave the field blank."
    )
    population_description: str = Field(
        description="Identify and extract population characteristics of participants who were approached to participate in this study. If available, provide an estimated number of the people or schools approached. Do not include inclusion or exclusion criteria, and baseline characteristics. Include information regarding people or other enrolled units."
    )
    setting: str = Field(
        description="List the country (or countries) in which the study was conducted. Describe the specific location setting (e.g. hospital, schools, outpatients, community). Be precise"
    )
    inclusion_criteria: str = Field(
        description="Identify and extract all inclusion criteria listed in the study. Note that inclusion criteria may differ for schools and students, so report each criterion individually and specify the group it applies to."
    )
    exclusion_criteria: str = Field(
        description="Identify and extract all exclusion criteria listed in the study. Note that exclusion criteria may differ for schools and students, so report each criterion individually and specify the group it applies to.  Don't extrapolate. Report answers as closely to the study text as possible. Leave the answer blank if not specified."
    )
    total_no_randomised: str = Field(
        description="If randomization was done, give the number of participants randomized at the baseline or first phase of the study, not necessarily the number who were actually treated or analyzed. If not, give the number of participants. If available, state the number per each trial arm in the study (e.g. control vs. intervention), together with the arm names, using bullet points, e.g. - Total: 296; - Control: 147; - Intervention: 149, and ensure that they add up to the total."
    )
    clusters: str = Field(
        description="If the study is a cluster randomized control trial, identify the type of cluster, number of clusters, number of individuals per cluster (provide a range if not given) and the total number of individuals. Leave blank if not a cluster randomized control trial."
    )
    age: str = Field(
        description="Give the mean and standard deviation (SD) of ages of participants who joined the study; provide either a mean, SD across all participants, or a mean, SD by group, or both. You may provide an age range if a mean is not available. Do not provide the age under inclusion or exclusion criteria'. Add a bullet point to categorise if the mean age falls between 0-5 years, 6-12 years, or 13-18 years."
    )
    sex: str = Field(
        description="Identify and extract the baseline characteristics of the sex or gender of participants. If the sex of participants is not explicitly mentioned, leave the answer blank. If more details are given on the gender distribution, you can answer with those details, e.g. 'mostly male' or '92% female and 8% male'. If multiple groups available, provide answer for all groups in bullet point. This answer is often presented in tables."
    )
    race_ethnicity: str = Field(
        description="Identify and extract any reported race or ethnicity in all groups. If the information is not reported in groups, provide an overall answer if available. This information is often presented in tables."
    )
    other_relevant_sociodemographics: str = Field(
        description="Identify and extract any reported sociodemographic. For example, socioeconomic status, socioeconomic index, low-income region, high-income region. Don't extrapolate based on income of country. Leave blank if not reported."
    )
    trial_registration_number: str = Field(
        description="Identify and extract the trial registration number of the study and the citation of any published protocol related to the study. Trial registrations typically begin with ANZCTR, ReBec, ChiCTR, CRiS, CTRI, RPCEC, EU-CTR, DRKS, IRCT, ISRCTN, ITMCTR, jRCT, LBCTR, TCTR, PACTR, REPEC, SLCTR. Published protocol related to the study are typically cited at the beginning of the methods with words such as 'for more detailed description, please see citation'."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study_author_contact_details",
            "aim_of_study",
            "design",
            "unit_of_allocation",
            "start_end_dates",
            "total_study_duration",
            "study_funding_sources",
            "possible_conflicts_of_interest",
            "population_description",
            "setting",
            "inclusion_criteria",
            "exclusion_criteria",
            "total_no_randomised",
            "clusters",
            "age",
            "sex",
            "race_ethnicity",
            "other_relevant_sociodemographics",
            "trial_registration_number",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "study_author_contact_details": self.study_author_contact_details,
            "aim_of_study": self.aim_of_study,
            "design": self.design,
            "unit_of_allocation": self.unit_of_allocation,
            "start_end_dates": self.start_end_dates,
            "total_study_duration": self.total_study_duration,
            "study_funding_sources": self.study_funding_sources,
            "possible_conflicts_of_interest": self.possible_conflicts_of_interest,
            "population_description": self.population_description,
            "setting": self.setting,
            "inclusion_criteria": self.inclusion_criteria,
            "exclusion_criteria": self.exclusion_criteria,
            "total_no_randomised": self.total_no_randomised,
            "clusters": self.clusters,
            "age": self.age,
            "sex": self.sex,
            "race_ethnicity": self.race_ethnicity,
            "other_relevant_sociodemographics": self.other_relevant_sociodemographics,
            "trial_registration_number": self.trial_registration_number,
        }


class Study(BaseModel):
    study_characteristics: Study_Characteristics = Field(
        description="Study-level metadata including design, population, setting, and key conclusions."
    )
    interventions: list[Intervention] = Field(
        description="All intervention groups (arms) identified in the trial."
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
