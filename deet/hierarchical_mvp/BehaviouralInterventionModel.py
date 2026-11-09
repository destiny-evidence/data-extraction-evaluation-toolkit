"""
Pydantic data models for behavioural-intervention (household energy) extraction.

The models mirror the hierarchy of the codebook:

    Study_Characteristics            study level
      EvaluationDesign               study level: how the effects are identified, outcomes
      Intervention                   one per treatment arm
        DeliveryDetails              only for Information / Feedback / Social Comparison arms
        PricingDetails               only for Monetary Incentives arms
        MotivationDetails            only for Motivation arms
      Candidate                      one per estimate listed in the document (no selection yet)
      CandidateRecord                candidate + the selection decision made in Python
      Effect                         selected candidate + its statistics + derived values

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
    EstimandAnalysis,
    EstimandAvailable,
    EstimandType,
    EstimateKind,
    EvaluationDesignType,
    ExclusionReason,
    FramingUnit,
    GeographicScope,
    InformationFrequency,
    InterventionSubtype,
    InterventionType,
    OutcomeScale,
    PeriodType,
    PopulationType,
    SignConvention,
    SignificanceReporting,
    SignificanceTest,
    StatisticalTechnique,
    StudyDesign,
    YesNo,
    YesNoUnclear,
)

# Output columns derived in Python (see derive_BI.py). Exported next to the reported columns.
DERIVED_FIELDS = (
    "derived_effect_direction",
    "derived_diff_mean",
    "derived_test_statistic",
    "derived_total_sample_size",
    "derived_significance_reporting",
    "derived_p_bound",
    "derived_percentage_change",
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
        description=(
            "Year(s) in which the fieldwork or intervention took place, not the publication year. "
            "Use the publication year only if no fieldwork or intervention date is given anywhere."
        )
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
            "Area in which the participating households live, not the location of the institution "
            "that ran the study. Stick to the listed options; describe anything smaller than a "
            "municipality in notes_study_scope."
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
        description=(
            "Yes if energy consumption was measured after the treatment had ended (follow-up). "
            "Measurements during the treatment, such as a mid-term survey, are not follow-up."
        )
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
# Study design level (one per document)
# ---------------------------------------------------------------------------


class Outcome(BaseModel):
    name: str = Field(
        description="Energy outcome as the study names it, e.g. 'electricity', 'gas', 'total energy'."
    )
    dependent_variable: str = Field(
        description="Energy-consumption measure with its unit in brackets, as in Effect_Stub.dependent_variable."
    )
    outcome_scale: OutcomeScale = Field(
        description="'log' if the outcome is a log, 'percent' if it is a percentage change, else 'absolute'."
    )


class EvaluationDesign(BaseModel):
    design: EvaluationDesignType = Field(
        description=(
            "How the effects are identified. RCT with control: random assignment to treatment and "
            "a control group. RCT between treatments only: random assignment, no untreated group. "
            "Quasi-experimental with comparison group: self-selected or matched comparison group. "
            "Natural experiment: exogenous change such as a price change or a regression "
            "discontinuity or kink. One-group pre-post: no comparison group."
        )
    )
    randomised: YesNo = Field(
        description=(
            "Yes if households were randomly assigned to treatment and control or between treatments. "
            "Quote the supporting sentence in randomisation_quote."
        )
    )
    randomisation_quote: str = Field(
        default="",
        description="Verbatim sentence on how households were assigned. Leave blank if none.",
    )
    estimand_available: EstimandAvailable = Field(
        description="ITT if an intention-to-treat estimate is reported; 'Not applicable' if there is no assignment."
    )
    outcomes: list[Outcome] = Field(
        description="Every energy outcome that the study estimates effects for."
    )
    primary_analysis: str = Field(
        description="Table or section where the main results are reported, e.g. 'Table 4'."
    )
    stated_preferred_specification: str = Field(
        description=(
            "Verbatim quote where the authors name a primary, preferred or main model. "
            "'not stated' if they do not."
        )
    )
    treatment_period: str = Field(
        description="Start and end of the treatment period (dates or months). 'NR' if not reported."
    )
    post_treatment_followup: YesNo = Field(
        description=(
            "Yes only if energy consumption was measured after the treatment had ended. "
            "Measurements during the treatment are not follow-up."
        )
    )
    notes_design: str = Field(
        default="",
        description="Remarks on the design that affect which estimates are relevant.",
    )


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
            "here; that is the study design. Use 'Difference of means' for t-tests or Welch tests "
            "of group means; use 'Random effects regression' for mixed or multilevel models."
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


class Candidate(Effect_Stub):
    """One estimate found in the document. Describes it; the selection is made in Python."""

    table_or_section: str = Field(
        description="Table, figure, appendix or section text where the estimate appears, e.g. 'Table 4' or 'Abstract'."
    )
    row_label: str = Field(
        description="Row label of the estimate in that table. Blank if not a table."
    )
    column_label: str = Field(
        description="Column or model label of the estimate. Blank if not a table."
    )
    outcome: str = Field(
        description="Name of the energy outcome from the evaluation design that this estimate measures."
    )
    estimand_type: EstimandType = Field(
        description=(
            "average_effect: the average effect of the intervention on the outcome. "
            "time_profile: an hourly or daily load-shape estimate. "
            "slope_or_elasticity: a price-response slope or elasticity. "
            "interaction_or_heterogeneity: an interaction term or a regression within the treated group. "
            "placebo_or_pre_trend: a pre-treatment or placebo term. "
            "descriptive_only: group means or a difference of means without a model."
        )
    )
    population: PopulationType = Field(
        description="full_sample if the estimate uses all participants; subgroup if it uses a part of them."
    )
    population_label: str = Field(
        default="",
        description="Name of the subgroup, e.g. 'low users'. Blank for full_sample.",
    )
    post_treatment_selected: YesNo = Field(
        description=(
            "Yes if the population was selected on behaviour after treatment started (e.g. households "
            "that played a game or completed a task). No otherwise."
        )
    )
    period: PeriodType = Field(
        description=(
            "full_treatment_period: the whole treatment period. sub_period: a part of it, such as "
            "3, 6 or 9 months after start. post_treatment_followup: after the treatment ended."
        )
    )
    period_label: str = Field(
        default="",
        description="Label of the sub-period, e.g. '6 months'. Blank otherwise.",
    )
    specification_label: str = Field(
        description="Model label, e.g. 'Model (3)' or 'OLS with household fixed effects'."
    )
    is_authors_preferred: YesNo = Field(
        description="Yes if the authors name this specification as primary, preferred or main (see the design's stated_preferred_specification)."
    )
    n_controls_and_fixed_effects: int = Field(
        default=0,
        description="Number of control variables plus fixed-effect sets in this specification. 0 for a base model.",
    )
    analysis: EstimandAnalysis = Field(
        description="Population the estimate refers to: ITT, ATT, LATE, Per-protocol or as-treated, or Unclear."
    )
    has_uncertainty: YesNo = Field(
        description="Yes if this estimate is reported with an SE, SD, CI, t-statistic, p-value or significance stars."
    )
    same_estimate_as: str = Field(
        default="",
        description=(
            "effect_number of another candidate that reports the same number elsewhere in the document "
            "(e.g. the abstract, or a table that repeats another table). Blank if none."
        ),
    )
    notes_candidate: str = Field(
        default="", description="Remarks on how the estimate was located or read."
    )


class CandidateRecord(BaseModel):
    """A candidate with the selection decision taken by select_candidates()."""

    candidate: Candidate
    selected: YesNo
    decision_reason: str


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
        default="",
        description="Remarks on how sample sizes relate to this estimate, e.g. whether N is the analysed N after attrition.",
    )


class SignificanceInfo(BaseModel):
    """Always asked, also when an SE is available, so the significance is recorded for every effect."""

    p_value: str = Field(
        description=(
            "Reported p-value, or the significance level (e.g. 0.05) if only a level is given. "
            "'NR' if not reported."
        )
    )
    stars: str = Field(
        default="",
        description="Significance stars printed next to the estimate, e.g. '**'. Blank if none are shown.",
    )
    star_legend: str = Field(
        default="",
        description="Star legend as printed in the table or its notes, e.g. '*** p<0.01, ** p<0.05, * p<0.1'. Blank if none.",
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


class EffectStatistics(BaseModel):
    """Output of the one-call statistics step for a selected candidate."""

    source_quote: str = Field(
        description=(
            "Verbatim text of the table row and column header, or of the sentence, that the numbers "
            "below come from. Copy it from the context. Fill this first."
        )
    )
    sign_convention: SignConvention = Field(
        description=(
            "'change in consumption' if the printed sign is the change in energy consumption (negative "
            "= consumption fell). 'reduction (positive = saving)' if the printed value is a saving, where "
            "a positive value means consumption fell."
        )
    )
    signed_point_estimate: str = Field(
        description="The point estimate with its sign exactly as printed (keep a minus sign). 'NR' if not reported."
    )
    coefficient_estimate: CoefficientEstimate | None = Field(
        default=None,
        description="Fill only when the estimate is a coefficient; leave empty otherwise.",
    )
    difference_of_means: DifferenceOfMeans | None = Field(
        default=None,
        description="Fill only when the estimate is a difference of means; leave empty otherwise.",
    )
    sample_sizes: SampleSizes = Field(
        description="Treatment, control and total sample sizes for this estimate."
    )
    significance: SignificanceInfo = Field(
        description="Significance information for this estimate. Always fill."
    )


class Effect(BaseModel):
    """One selected estimate: the candidate plus its statistics and the Python-derived values."""

    candidate: Candidate
    source_quote: str = ""
    sign_convention: str = ""
    signed_point_estimate: str = ""
    coefficient_estimate: CoefficientEstimate | None = None
    difference_of_means: DifferenceOfMeans | None = None
    sample_sizes: SampleSizes | None = None
    significance: SignificanceInfo | None = None
    baseline: BaselineInfo | None = None
    # Keys are DERIVED_FIELDS; computed deterministically in derive_BI.py, never by the LLM.
    derived: dict[str, str] = Field(default_factory=dict)
    notes_effect: str = Field(
        default="",
        description="Remarks on how this estimate was obtained or interpreted, including derived-value mismatches.",
    )
    selection_reason: str = Field(
        default="",
        description="Why this candidate was selected. Set by select_candidates(), not queried.",
    )


# ---------------------------------------------------------------------------
# Top-level Study model
# ---------------------------------------------------------------------------


class Study(BaseModel):
    screening: Screening
    study_characteristics: Study_Characteristics | None = None
    evaluation_design: EvaluationDesign | None = None
    interventions: list[Intervention] = Field(default_factory=list)
    arm_details: list[Arm_Details] = Field(default_factory=list)
    candidates: list[CandidateRecord] = Field(default_factory=list)
    effects: list[Effect] = Field(default_factory=list)
    design_reasoning: str = Field(
        default="",
        description="Reasoning of the design step. Set by the pipeline, not queried.",
    )
    listing_reasoning: str = Field(
        default="",
        description="Reasoning of the candidate-listing step. Set by the pipeline, not queried.",
    )

    # -- CSV export ---------------------------------------------------------
    # Four tables joined by `study` (and `arm`): study, interventions, candidates, effects.

    @classmethod
    def csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Screening),
            *_field_names(Study_Characteristics, skip={"study"}),
            *_field_names(EvaluationDesign, skip={"outcomes"}),
            "outcomes",
            "design_reasoning",
            "listing_reasoning",
        ]

    def to_csv_row(self, study_id: str = "", external_id: str = "") -> dict[str, str]:
        row = {"study": self._study_id(study_id), "external_id": external_id}
        row.update(_flatten(self.screening))
        if self.study_characteristics is not None:
            row.update(_flatten(self.study_characteristics, skip={"study"}))
        if self.evaluation_design is not None:
            row.update(_flatten(self.evaluation_design, skip={"outcomes"}))
            row["outcomes"] = _outcomes_text(self.evaluation_design.outcomes)
        row["design_reasoning"] = self.design_reasoning
        row["listing_reasoning"] = self.listing_reasoning
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
    def candidate_csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Candidate),
            "selected",
            "decision_reason",
        ]

    def candidate_csv_rows(
        self, study_id: str = "", external_id: str = ""
    ) -> list[dict[str, str]]:
        study_id = self._study_id(study_id)
        return [
            {
                "study": study_id,
                "external_id": external_id,
                **_flatten(record.candidate),
                "selected": record.selected,
                "decision_reason": record.decision_reason,
            }
            for record in self.candidates
        ]

    @classmethod
    def outcome_csv_fieldnames(cls) -> list[str]:
        return [
            "study",
            "external_id",
            *_field_names(Candidate),
            "source_quote",
            "sign_convention",
            "signed_point_estimate",
            *_field_names(CoefficientEstimate),
            *_field_names(DifferenceOfMeans, skip=_SHARED),
            *_field_names(SampleSizes),
            *_field_names(SignificanceInfo),
            *_field_names(BaselineInfo),
            *DERIVED_FIELDS,
            "notes_effect",
            "selection_reason",
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
                **_flatten(effect.candidate),
                "source_quote": effect.source_quote,
                "sign_convention": effect.sign_convention,
                "signed_point_estimate": effect.signed_point_estimate,
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
            row.update(effect.derived)
            row["notes_effect"] = effect.notes_effect
            row["selection_reason"] = effect.selection_reason
            rows.append(row)
        return rows

    def _study_id(self, fallback: str) -> str:
        if self.study_characteristics is not None:
            return self.study_characteristics.study.strip() or fallback
        return fallback


_SHARED = {"effect_direction"}


def _outcomes_text(outcomes: list[Outcome]) -> str:
    return "; ".join(
        f"{o.name} ({o.dependent_variable}; {o.outcome_scale})" for o in outcomes
    )


def _field_names(model: type[BaseModel], skip: set[str] = frozenset()) -> list[str]:
    return [name for name in model.model_fields if name not in skip]


def _flatten(model: BaseModel, skip: set[str] = frozenset()) -> dict[str, str]:
    """Flat dict of a model's fields; lists are joined with '; '."""
    return {
        name: "; ".join(value) if isinstance(value, list) else value
        for name, value in model.model_dump().items()
        if name not in skip
    }
