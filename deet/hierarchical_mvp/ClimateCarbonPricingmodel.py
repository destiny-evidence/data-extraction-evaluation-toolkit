from __future__ import annotations

from pydantic import BaseModel, Field


class Intervention(BaseModel):
    intervention_name: str = Field(
        description="Provide the name of the carbon pricing intervention."
    )
    type_of_intervention: str = Field(
        description="The type of the carbon pricing intervention within this policy."
    )
    name: str = Field(
        description="Name of the carbon pricing policy that this intervention is a part of. Leave blank if not reported."
    )
    start_end_date: str = Field(
        description="The start and end date of the intervention. In a synthetic control study, capture the first year of the period on which the calculation of the weights is based on. Use format DD.MM.YYYY - DD.MM.YYYY and leave blank if not reported."
    )
    geographicLocationTreatmentGroup: str = Field(
        description='Capture the geographic location for which emission data was analysed. Here we do not look for the scope of the intervention. For the effect of the EU ETS on emission from the German electricity sector the value will be Germany. Note: Try and stick to the format "city/state;country/block". In case that there is more than one entity analysed connect them with an & i.e. "city/state;country/block&city/state;country/block". Remove all white spaces. If country names require a whitespace, replace it with a lower bar. Leave blank if not reported.'
    )
    emission_sector: str = Field(
        description="Capture the emission sector which the analysis is performed on. Choose one or multiple options. Note: if sectors are not specified but it is indicated that all sectors covered by an intervention are analysed choose all covered sectors. [Energy, Industry, Transport, Buildings, AFOLU, International Aviation and Shipping, Economy, All covered sectors]. Leave blank if not reported."
    )
    carbon_price: str = Field(
        description="Extract information about the carbon price under this policy. Leave blank if not reported."
    )
    Fuel_type: str = Field(
        description='Extract the fuel type that is covered by the intervention, leave blank if not reported or extract "All fuels" if it is stated explicitly or if it is likely that everything is covered. Otherwise, if applicable, choose one or more from this list: [Coal, Natural gas, Petrol, Gasoline, Diesel]'
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "intervention_name",
            "type_of_intervention",
            "name",
            "start_end_date",
            "geographicLocationTreatmentGroup",
            "emission_sector",
            "carbon_price",
            "Fuel_type",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "intervention_name": self.intervention_name,
            "type_of_intervention": self.type_of_intervention,
            "name": self.name,
            "start_end_date": self.start_end_date,
            "geographicLocationTreatmentGroup": self.geographicLocationTreatmentGroup,
            "emission_sector": self.emission_sector,
            "carbon_price": self.carbon_price,
            "Fuel_type": self.Fuel_type,
        }


class Effect_Outcome(BaseModel):
    effect_name: str = Field(
        description="The name of the effect/outcome for which data are extracted."
    )
    intervention_name: str = Field(
        description="The name of the intervention that produced the outcome being measured here."
    )
    dependent_variable_co2_ghg: str = Field(
        description="Which of these is a dependent variable for this outcome, choose either [CO2, Greenhouse gases (in CO2 equivalents)]"
    )
    dependent_variable_total_capita: str = Field(
        description="Which of these is a dependent variable for this outcome, choose either [Total, Per capita, Per unit of GDP or output]"
    )
    transformations: str = Field(
        description="Capture whether the dependent and/or independent variables are transformed to log values. e.g., choose log-level if the dependent variable is in logs and the independent variable is in levels. Choose one of the following: [Level-level, Log-Level, Log-Log, Level-Log, Relative change in percent/100]. Leave blank if not reported."
    )
    effectSize_statisticalEstimate: str = Field(
        description="Capture the statistical estimate of the effect of the explanatory variable (introduction/existence of carbon pricing/carbon price) on the outcome variable (emissions etc.) as given by the authors."
    )
    standardError: str = Field(
        description="If provided, extract the standard error for this outcome. Otherwise, leave the field blank."
    )
    t_statistic: str = Field(
        description="If provided, extract the t-statistic value for this outcome. Otherwise, leave the field blank."
    )
    p_value: str = Field(
        description="If provided, extract the p-value for this outcome. Otherwise, leave the field blank."
    )
    confidence_interval_bounds: str = Field(
        description="If provided, extract the confidence interval (upper and lower bounds) for this outcome. Otherwise, leave the field blank."
    )
    sample_size: str = Field(
        description="Capture the observed entities/individuals with as much detail as possible (i.e. if all three options are given, record all). 1. Total – Full sample size (for synthetic control estimations, capture the treatment group and the full donor pool). 2. Treatment – Capture sample size of entities with carbon pricing. 3. Control – Capture the sample size of entities without carbon pricing. Leave blank if not reported."
    )
    info_location: str = Field(
        description="The location within the source text from where the information about statistical data (effectSize_statisticalEstimate, transformations, standardError, t_statistic, p_value, confidence_interval_bounds, sample size) as applicable was extracted, e.g Section, Table or Figure number."
    )
    significance: str = Field(
        description="If the study indicates the level of significance at which the effect size is significant, capture the highest level of indicated significance. Often this indication of significance is provided by asterisks/stars in the results table or it is mentioned in the text. Choose one of the following: [<0.90, 0.90, 0.95, 0.99, >0.99, Insignificant, not reported]"
    )
    effect_size_direction: str = Field(
        description='Capture the direction of the effect carbon pricing had on the outcome variable. If the effect suggests that carbon pricing decreased the outcome variable relative to a baseline/control scenario choose "decrease". Otherwise choose "increase"'
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "effect_name",
            "intervention_name",
            "dependent_variable_co2_ghg",
            "dependent_variable_total_capita",
            "transformations",
            "effectSize_statisticalEstimate",
            "standardError",
            "t_statistic",
            "p_value",
            "confidence_interval_bounds",
            "sample_size",
            "info_location",
            "significance",
            "effect_size_direction",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "effect_name": self.effect_name,
            "intervention_name": self.intervention_name,
            "dependent_variable_co2_ghg": self.dependent_variable_co2_ghg,
            "dependent_variable_total_capita": self.dependent_variable_total_capita,
            "transformations": self.transformations,
            "effectSize_statisticalEstimate": self.effectSize_statisticalEstimate,
            "standardError": self.standardError,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "confidence_interval_bounds": self.confidence_interval_bounds,
            "sample_size": self.sample_size,
            "info_location": self.info_location,
            "significance": self.significance,
            "effect_size_direction": self.effect_size_direction,
        }


class Study_Characteristics(BaseModel):
    design_plain: str = Field(
        description="Provide a concise description of the study design used assess the effectiveness of carbon pricing in reducing emissions."
    )
    design_category: str = Field(
        description="Chose the most appropriate design for this study from the following list: [Difference-in-Differences (DiD), Triple Differences (DiDiD or DDD), OLS (pure cross section), Machine Learning, Time Series, Synthetic Control, Unclear, Not reported]."
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "design_plain",
            "design_category",
        ]

    def to_csv_row(self) -> dict[str, str]:
        return {
            "design_plain": self.design_plain,
            "design_category": self.design_category,
        }


class Study(BaseModel):
    study_characteristics: Study_Characteristics = Field(
        description="Study-level metadata including the study design used to assess the effectiveness of carbon pricing."
    )
    interventions: list[Intervention] = Field(
        description="All carbon pricing interventions identified in the study."
    )
    effect_outcomes: list[Effect_Outcome] = Field(
        default_factory=list,
        description="All effect/outcome data extracted from the study, reporting the estimated effect of carbon pricing on emissions.",
    )

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return Study_Characteristics.csv_fieldnames()

    def to_csv_row(self) -> dict[str, str]:
        return self.study_characteristics.to_csv_row()

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        return Effect_Outcome.csv_fieldnames()
