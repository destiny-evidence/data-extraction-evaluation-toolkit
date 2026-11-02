"""Pydantic data models for behavioural-intervention (household energy) extraction.

The models mirror the hierarchy of the codebook:

    Study_Characteristics            study level
      Intervention                   one per treatment arm
        DeliveryDetails              only for Information / Feedback / Social Comparison arms
        PricingDetails               only for Monetary Incentives arms
        MotivationDetails            only for Motivation arms
      Effect_Stub                    one per reported estimate (where / which arms / which kind)
      Effect                         stub + the estimate block selected by the stub

Sub-models that are only queried when a higher level calls for them are held as
Optional: ``None`` means "not applicable / not queried", while the string "NR"
means "queried but not reported in the paper".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from deet.logger import logger

from .vocab_BI import (
    BaselineSource,
    CoefficientSdType,
    ControlVariable,
    DocumentType,
    EffectDirection,
    EstimateKind,
    ExclusionReason,
    FramingUnit,
    GeographicScope,
    InformationFrequency,
    InterventionSubtype,
    InterventionType,
    OutcomeScale,
    SignificanceReporting,
    SignificanceTest,
    StatisticalTechnique,
    StudyDesign,
    YesNo,
    YesNoUnclear,
)

# ---------------------------------------------------------------------------
# Study level
# ---------------------------------------------------------------------------


class Screening(BaseModel):
    exclusion_reason: ExclusionReason = Field(
        description=(
            "Leave empty if the document can be coded. Otherwise give the reason it must be "
            "excluded: it contains no relevant effect sizes, the effect size information is "
            "incomplete or irretrievable, or the document is not about a behavioural "
            "intervention targeting household energy use."
        )
    )
    notes_exclusion_reason: str = Field(
        default="",
        description="Short explanation of the exclusion reason. Leave blank if the document is included.",
    )


class Study_Characteristics(BaseModel):
    study: str = Field(
        description=(
            "Unique study identifier in 'first author-year' format (e.g. 'Smith-2021'). "
            "This value must be consistent across all extracted rows for this study."
        )
    )
    authors: str = Field(
        description="All authors, separated by '; ' (e.g. 'Lee, M; Kim, D')."
    )
    title: str = Field(description="Full title of the document.")
    journal: str = Field(
        default="",
        description="Journal or publication venue. Leave blank if not reported.",
    )
    doi: str = Field(
        default="", description="DOI of the document. Leave blank if not reported."
    )
    publication_year: str = Field(description="Year the document was published.")
    study_year: str = Field(
        description="Year in which the study (intervention) took place; use the publication year if not stated."
    )
    document_type: DocumentType = Field(
        description=(
            "Peer-Reviewed Paper: academic paper published in a peer-reviewed journal. "
            "Unpublished Academic Paper: working, discussion or similar paper not published in a "
            "peer-reviewed journal. Conference Paper: conference proceedings."
        )
    )
    geographic_scope: GeographicScope = Field(
        description=(
            "Area across which the intervention was carried out. Stick to the listed options; "
            "describe anything smaller than a municipality in notes_study_scope."
        )
    )
    city: str = Field(
        description="City of the study in 'city' form. Leave blank if not reported."
    )
    state: str = Field(
        description="State/province/region of the study. Leave blank if not reported."
    )
    country: str = Field(
        description="Country of the study. Leave blank if not reported."
    )
    notes_study_scope: str = Field(
        default="",
        description="Describe the study area if it is smaller than a municipality (e.g. 'dorms in a university').",
    )
    randomisation_dummy: YesNo = Field(
        description=(
            "Yes if households were randomly assigned to control and treatment groups and/or "
            "between different treatment groups, otherwise No."
        )
    )
    opt_in_out: YesNo = Field(
        description=(
            "Yes if households were required to explicitly opt in to the intervention/experiment. "
            "No if they were selected and could opt out, if participation was mandatory, or if "
            "there was neither."
        )
    )
    smart_metering: YesNo = Field(
        description="Yes if smart meters were used to measure energy consumption, otherwise No."
    )
    notes_doc_level: str = Field(
        default="",
        description="Any other noteworthy observation about the study as a whole.",
    )


# ---------------------------------------------------------------------------
# Intervention level (one per treatment arm)
# ---------------------------------------------------------------------------


class Intervention(BaseModel):
    arm: str = Field(
        description=(
            "Name of this treatment arm exactly as labelled in the study (e.g. 'Treatment 1', "
            "'Peer comparison + tips'). Control groups are not interventions."
        )
    )
    description: str = Field(
        description="Description of what households in this arm received, in as much detail as the study gives."
    )
    intervention_types: list[InterventionType] = Field(
        description=(
            "All broad types this arm belongs to. Information: home audits, tips. "
            "Feedback: historical, enhanced billing, in-home display. "
            "Social Comparison: peer comparison, home energy report. "
            "Monetary Incentives: rewards, ToU, real-time, dynamic, peak load pricing. "
            "Motivation: gamification, commitments, goal setting. "
            "A joint intervention has several types."
        )
    )
    intervention_subtypes: list[InterventionSubtype] = Field(
        description="All sub-types that are part of this arm, consistent with intervention_types."
    )
    additional_intervention: str = Field(
        default="",
        description="Any additional component that fits none of the sub-types. Leave blank if none.",
    )
    intervention_duration_weeks: str = Field(
        description=(
            "Total duration of the experiment in weeks, including baseline, treatment and follow-up "
            "periods. Use 1 month = 4 weeks and 1 year = 52 weeks. For a pre/post set-up where the "
            "control period is the same months of the previous year, count only the treatment-length "
            "period (e.g. 3 months -> 12). Leave blank if not reported."
        )
    )
    treatment_period_weeks: str = Field(
        description=(
            "Duration of the treatment itself in weeks (1 month = 4 weeks, 1 year = 52 weeks). "
            "If two separate periods are pooled, give the sum. Leave blank if not reported."
        )
    )
    followup: YesNoUnclear = Field(
        description="Yes if energy consumption was investigated in a follow-up after the treatment, otherwise No."
    )
    notes_duration: str = Field(
        default="", description="Remarks on the periods (e.g. pre-intervention weeks)."
    )
    notes_intervention_type: str = Field(
        default="", description="Remarks on how the arm was coded by type."
    )


class DeliveryDetails(BaseModel):
    """Asked only for arms that contain Information, Feedback or Social Comparison."""

    framing_unit: FramingUnit = Field(
        description="Units/terms in which consumption information or feedback is given to households."
    )
    information_frequency: InformationFrequency = Field(
        description="Frequency at which households are contacted with information or feedback."
    )
    medium: str = Field(
        description="How households were contacted (e.g. mail, email, app, in-home display, home visit)."
    )


class PricingDetails(BaseModel):
    """Asked only for arms that contain Monetary Incentives."""

    price_scheme: str = Field(
        description="Describe the pricing scheme or reward (e.g. peak/off-peak prices, reward per kWh saved)."
    )
    notification_medium: str = Field(
        default="",
        description="How households were told about prices or rewards. Leave blank if not reported.",
    )


class MotivationDetails(BaseModel):
    """Asked only for arms that contain Motivation."""

    motivation_mechanism: str = Field(
        description="Describe the mechanism (e.g. pledge, savings goal of 10%, game with points)."
    )
    goal_or_commitment: str = Field(
        default="",
        description="The goal level or commitment households made, if any. Leave blank if not reported.",
    )


class Arm_Details(BaseModel):
    """Container for the optional per-type sub-models of one arm."""

    arm: str = Field(description="Arm label this detail block belongs to.")
    delivery: DeliveryDetails | None = None
    pricing: PricingDetails | None = None
    motivation: MotivationDetails | None = None


# ---------------------------------------------------------------------------
# Effect level
# ---------------------------------------------------------------------------


class Effect_Stub(BaseModel):
    """Where an estimate is and what kind it is; decides which fields are asked next."""

    effect_number: str = Field(
        description=(
            "Number identifying the estimate; larger numbers for estimates that appear later in the "
            "text. Need not start at 1."
        )
    )
    page: str = Field(
        description=(
            "Page number of the table presenting the estimate. If the document has no page numbers, "
            "use a normal page count."
        )
    )
    estimate_label: str = Field(
        description=(
            "Exact location of the estimate, specific enough to tell it apart from every other "
            "estimate: table or section, plus the row/variable and the column/model label "
            "(e.g. 'Table 4, row \"Treatment\", model (2)')."
        )
    )
    intervention_arms: list[str] = Field(
        description="Labels of the treatment arm(s) this estimate refers to (several for a joint intervention)."
    )
    control_description: str = Field(
        description=(
            "'No treatment' if the arm is compared to no treatment, otherwise the arm/configuration "
            "that serves as the control."
        )
    )
    study_design: StudyDesign = Field(
        description=(
            "Pre-test/post-test: no control, same group measured twice. "
            "Control-treatment: no baseline, two groups observed starting at the same time. "
            "Difference in Difference: baseline for both control and treatment groups, then treatment "
            "and measurement of the effect."
        )
    )
    statistical_technique: StatisticalTechnique = Field(
        description=(
            "Type of regression/test used for this estimate. Do NOT code Difference in Difference "
            "here; that is the study design."
        )
    )
    dependent_variable: str = Field(
        description=(
            "Energy-consumption measure the authors use, with its unit in brackets "
            "(e.g. 'household hourly electricity consumption (kWh)'). Intention-to-treat estimates "
            "only; skip ATT/LATE estimates."
        )
    )
    outcome_scale: OutcomeScale = Field(
        description="'log' if the dependent variable is a log, 'percent' if it is a percentage change, else 'absolute'."
    )
    estimate_kind: EstimateKind = Field(
        description=(
            "'coefficient' if a regression coefficient is reported for this estimate, "
            "'difference_of_means' if group means (or a difference of means) are reported."
        )
    )


class CoefficientEstimate(BaseModel):
    """Asked only when the stub's estimate_kind is 'coefficient'."""

    coefficient: str = Field(
        description="Absolute value of the relevant regression coefficient. 'NR' if not reported."
    )
    effect_direction: EffectDirection = Field(
        description="Increase if energy consumption increased after the intervention (undesired), Decrease if it fell."
    )
    coefficient_sd: str = Field(
        description="Standard error (or SD) of the coefficient as reported. 'NR' if not reported."
    )
    coefficient_sd_type: CoefficientSdType = Field(
        description="Type of uncertainty measure provided."
    )
    controls: list[ControlVariable] = Field(
        description=(
            "Explicit control variables in the model. Household or time fixed/random effects are NOT "
            "controls (they belong in statistical_technique). If the baseline is weather-corrected "
            "in a pre/post analysis, include Weather controls. Empty list for a base model."
        )
    )


class DifferenceOfMeans(BaseModel):
    """Asked only when the stub's estimate_kind is 'difference_of_means'."""

    control_mean: str = Field(
        description="Mean of the control group. 'NR' if not reported."
    )
    control_sd: str = Field(
        description="SD of the control group. 'NR' if not reported."
    )
    treated_mean: str = Field(
        description="Mean of the treated group. 'NR' if not reported."
    )
    treated_sd: str = Field(
        description="SD of the treated group. 'NR' if not reported."
    )
    diff_mean: str = Field(
        description="Reported difference of means. 'NR' if not reported."
    )
    effect_direction: EffectDirection = Field(
        description="Increase if energy consumption increased after the intervention (undesired), Decrease if it fell."
    )


class SampleSizes(BaseModel):
    """Sample sizes for one estimate; which fields are asked depends on the study design."""

    total_sample_size: str = Field(
        description=(
            "Control plus all relevant treatments, full sample size. If the control size is not "
            "clearly specified, be conservative and use the treatment sample size. 'NR' if not reported."
        )
    )
    treatment_sample_size: str = Field(
        default="",
        description="Sample size of the specific treatment (or combination) for this estimate. 'NR' if not reported.",
    )
    control_sample_size: str = Field(
        default="",
        description="Sample size of the control group (possibly shared by several treatments). 'NR' if not reported.",
    )
    notes_sample_size: str = Field(
        default="", description="Remarks on how sample sizes relate to this estimate."
    )


class SignificanceInfo(BaseModel):
    """Asked only when no standard error/SD is available to compute the test statistic."""

    p_value: str = Field(
        description=(
            "Reported p-value, or the significance level (e.g. 0.05) if only a level is given. "
            "'NR' if not reported."
        )
    )
    significance_reporting: SignificanceReporting = Field(
        description=(
            "'actual' if an exact p-value or test statistic is given, 'upper bound' if only "
            "significance at a level is given for a significant effect, 'lower bound' if only "
            "'not significant' is given, 'none' if nothing is reported."
        )
    )
    significance_test: SignificanceTest = Field(
        description="Test behind the significance statement."
    )
    test_statistic: str = Field(
        description="Reported test statistic (t, F, z, Chi). 'NR' if not reported."
    )
    test_tails: str = Field(
        description="'1' or '2' for one- or two-tailed. 'NR' if not reported."
    )
    notes_significance: str = Field(
        default="", description="Remarks on the significance information."
    )


class BaselineInfo(BaseModel):
    """Asked only when the dependent variable is in absolute units (so a % change can be derived)."""

    baseline_consumption: str = Field(
        description=(
            "Average consumption in energy units during the period the coefficient/difference refers "
            "to. 'NR' if not reported."
        )
    )
    baseline_source: BaselineSource = Field(
        description=(
            "Which average was recorded. Preference order: (1) all households in the baseline period, "
            "(2) weighted treatment and control in the baseline period, (3) all households in the "
            "treatment period, (4) weighted treatment and control in the treatment period, "
            "(5) control group. Use the highest option available."
        )
    )
    notes_baseline: str = Field(
        default="", description="Remarks on what the baseline consumption refers to."
    )


class Effect(BaseModel):
    """One estimate: the stub plus the blocks the stub required."""

    stub: Effect_Stub
    coefficient_estimate: CoefficientEstimate | None = None
    difference_of_means: DifferenceOfMeans | None = None
    sample_sizes: SampleSizes | None = None
    significance: SignificanceInfo | None = None
    baseline: BaselineInfo | None = None
    notes_effect: str = Field(
        default="",
        description="Remarks on how this estimate was obtained or interpreted.",
    )
    inclusion_reasoning: str = Field(
        default="",
        description=(
            "Reasoning of the effect-listing step on why the estimates were included. Set by the "
            "pipeline, not queried from the LLM."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level Study model
# ---------------------------------------------------------------------------


class Study(BaseModel):
    screening: Screening
    study_characteristics: Study_Characteristics | None = None
    interventions: list[Intervention] = Field(default_factory=list)
    arm_details: list[Arm_Details] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)

    # -- CSV export ---------------------------------------------------------
    # Three tables joined by `study` (and `arm`): study, interventions, effects.

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Screening),
            *_field_names(Study_Characteristics, skip={"study"}),
        ]

    def to_csv_row(
        self, study_id: str = "", external_id: str = ""
    ) -> dict[str, str]:
        row = {"study": self._study_id(study_id), "external_id": external_id}
        row.update(_flatten(self.screening))
        if self.study_characteristics is not None:
            row.update(_flatten(self.study_characteristics, skip={"study"}))
        return row

    @classmethod
    def intervention_csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Intervention),
            *_field_names(DeliveryDetails),
            *_field_names(PricingDetails),
            *_field_names(MotivationDetails),
        ]

    def intervention_csv_rows(
        self, study_id: str = "", external_id: str = ""
    ) -> list[dict[str, str]]:
        study_id = self._study_id(study_id)
        details = {d.arm: d for d in self.arm_details}
        unmatched = set(details) - {arm.arm for arm in self.interventions}
        if unmatched:
            logger.warning(
                f"arm_details without a matching intervention were not exported: "
                f"{sorted(unmatched)}"
            )
        rows = []
        for arm in self.interventions:
            row = {"study": study_id, "external_id": external_id, **_flatten(arm)}
            if (detail := details.get(arm.arm)) is not None:
                for block in (detail.delivery, detail.pricing, detail.motivation):
                    if block is not None:
                        row.update(_flatten(block))
            rows.append(row)
        return rows

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Effect_Stub),
            *_field_names(CoefficientEstimate),
            *_field_names(DifferenceOfMeans, skip=_SHARED),
            *_field_names(SampleSizes),
            *_field_names(SignificanceInfo),
            *_field_names(BaselineInfo),
            "notes_effect",
            "inclusion_reasoning",
        ]

    def outcome_csv_rows(
        self, study_id: str = "", external_id: str = ""
    ) -> list[dict[str, str]]:
        study_id = self._study_id(study_id)
        rows = []
        for effect in self.effects:
            row = {
                "study": study_id,
                "external_id": external_id,
                **_flatten(effect.stub),
            }
            # effect_direction is shared by both estimate kinds: one column.
            for block in (
                effect.coefficient_estimate,
                effect.difference_of_means,
                effect.sample_sizes,
                effect.significance,
                effect.baseline,
            ):
                if block is not None:
                    row.update(_flatten(block))
            row["notes_effect"] = effect.notes_effect
            row["inclusion_reasoning"] = effect.inclusion_reasoning
            rows.append(row)
        return rows

    def _study_id(self, fallback: str) -> str:
        if self.study_characteristics is not None:
            return self.study_characteristics.study.strip() or fallback
        return fallback


_SHARED = {"effect_direction"}


def _field_names(model: type[BaseModel], skip: set[str] = frozenset()) -> list[str]:
    return [name for name in model.model_fields if name not in skip]


def _flatten(model: BaseModel, skip: set[str] = frozenset()) -> dict[str, str]:
    """Flat dict of a model's fields; lists are joined with '; '."""
    return {
        name: "; ".join(value) if isinstance(value, list) else value
        for name, value in model.model_dump().items()
        if name not in skip
    }
