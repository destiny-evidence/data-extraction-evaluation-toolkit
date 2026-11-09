"""
DSPy signatures and gated extraction pipeline for behavioural interventions
targeting household energy use.

Same structure as the Cochrane RCT pipeline, but each level is only queried when
the level above calls for it (gating is plain Python in ``forward``, not LLM-decided):

  L0  ScreenDocument               -> stop if the document is excluded
  L1  ExtractStudyInfo             -> stop if no treatment arm is found
  L2  Extract*Details              -> only for the broad types an arm actually has
  L3a ClassifyEvaluationDesign     -> how the effects are identified (one per document)
  L3b ListCandidateEstimates       -> every estimate, described but not selected
  L3c select_candidates()          -> which estimates to code (plain Python, selection_BI.py)
  L3d ExtractEffectStatistics      -> one call per selected estimate, always with significance
  L3e derive()                     -> derived values in Python (derive_BI.py)
"""

from __future__ import annotations

import dspy

from deet.logger import logger

from .BehaviouralInterventionModel import (
    Arm_Details,
    BaselineInfo,
    Candidate,
    DeliveryDetails,
    Effect,
    EffectStatistics,
    EvaluationDesign,
    Intervention,
    MotivationDetails,
    PricingDetails,
    Screening,
    Study,
    Study_Characteristics,
)
from .derive_BI import derive
from .selection_BI import select_candidates

_NOT_REPORTED = {"", "NR", "NA"}

_CONTEXT_DESC = (
    "Concatenated markdown text from one or more parsed PDFs, all describing the same "
    "behavioural-intervention study"
)

# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


class ScreenDocument(dspy.Signature):
    """
    You are a systematic review assistant for a review of behavioural interventions that
    reduce household energy consumption.

    Decide whether the document can be coded. It can be coded if it reports effect sizes
    of an intervention on household energy consumption with enough information to
    retrieve them.
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    screening: Screening = dspy.OutputField(
        desc="Whether the document is excluded, and why"
    )


class ExtractStudyInfo(dspy.Signature):
    """
    You are a systematic review assistant for a review of behavioural interventions that
    reduce household energy consumption.

    Extract the study-level metadata and identify every distinct treatment arm. An arm is a
    distinct intervention configuration that households received (a combination of
    interventions is one arm). Do not list control groups as interventions.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    study_characteristics: Study_Characteristics = dspy.OutputField(
        desc="Study-level metadata and design characteristics"
    )
    interventions: list[Intervention] = dspy.OutputField(
        desc="Every treatment arm, classified by broad type and sub-type"
    )


class ExtractDeliveryDetails(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given treatment arm, which contains information, feedback or social comparison
    elements, extract how the information was delivered.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    intervention: Intervention = dspy.InputField(desc="The treatment arm to describe")
    delivery: DeliveryDetails = dspy.OutputField(
        desc="Framing, frequency and medium of the information"
    )


class ExtractPricingDetails(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given treatment arm, which contains a monetary incentive (pricing or reward),
    extract the details of the scheme.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    intervention: Intervention = dspy.InputField(desc="The treatment arm to describe")
    pricing: PricingDetails = dspy.OutputField(
        desc="Details of the pricing scheme or reward"
    )


class ExtractMotivationDetails(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given treatment arm, which contains a motivational element (gamification,
    commitment, goal setting), extract the details of the mechanism.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    intervention: Intervention = dspy.InputField(desc="The treatment arm to describe")
    motivation: MotivationDetails = dspy.OutputField(
        desc="Details of the motivational mechanism"
    )


class ClassifyEvaluationDesign(dspy.Signature):
    """
    You are a systematic review assistant for a review of behavioural interventions that
    reduce household energy consumption.

    Describe how this study identifies the effect of the intervention on household energy
    consumption: the design, whether households were randomly assigned, which energy outcomes
    are estimated, which specification the authors name as primary, and whether energy
    consumption was measured after the treatment had ended.

    Follow-up means measurement after the treatment ended. Estimates made during the
    treatment, such as a mid-term survey or a check after a few months of treatment, are not
    follow-up.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    interventions: list[Intervention] = dspy.InputField(
        desc="The treatment arms already identified"
    )
    evaluation_design: EvaluationDesign = dspy.OutputField(
        desc="Design, randomisation, outcomes, primary analysis and follow-up of the study"
    )


class ListCandidateEstimates(dspy.Signature):
    """
    You are a systematic review assistant.

    List every estimate of the effect of an intervention on household energy consumption
    that the study reports. An estimate qualifies if it contrasts energy consumption under
    the intervention with a counterfactual: treated versus control households, or the same
    households before versus during/after the intervention. List every such estimate, including
    alternative specifications, sub-periods, subgroups and descriptive contrasts. A later step
    selects which ones to code, so do not leave an estimate out because it may not be selected.
    Describe each estimate with its attributes (location, arms, population, period,
    specification, analysis and whether it reports uncertainty).

    Do NOT list:
    - coefficients of covariates or other explanatory variables that are not a treatment
      indicator (e.g. household characteristics, attitudes, or changes in behaviour);
    - the same number more than once. If an estimate is reported in several places (abstract,
      text, table), list it once from the location that also reports its uncertainty, and
      point same_estimate_as at the other entry if you list both.

    In your reasoning, go through the estimates by effect_number and describe the attributes
    that matter for the selection (population, period, analysis, specification, uncertainty).

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    interventions: list[Intervention] = dspy.InputField(
        desc="The treatment arms already identified"
    )
    evaluation_design: EvaluationDesign = dspy.InputField(
        desc="How the study identifies the effect, and its outcomes"
    )
    candidates: list[Candidate] = dspy.OutputField(
        desc="One entry per reported estimate, with its location, arms, population, period and specification"
    )


class ExtractEffectStatistics(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract its statistics: the point estimate with its sign, the
    coefficient or the group means and differences, the uncertainty, the sample sizes and the
    significance. Start by copying the table row and column header (or the sentence) the numbers
    come from into source_quote.

    Fill coefficient_estimate when the estimate is a regression coefficient (estimate_kind
    'coefficient'); fill difference_of_means when it is a difference or group means
    (estimate_kind 'difference_of_means'). For a pre-test/post-test design with no control group,
    report only the total sample size and leave the treatment and control sizes as NR.

    Always extract the significance: stars and their legend, the p-value, the test, the test
    statistic and the number of tails, whether or not an SE is reported.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source, including their sign; do not calculate or impute. If a value
    is not reported, use the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Candidate = dspy.InputField(
        desc="The estimate to extract (see its table, row, column and page)"
    )
    statistics: EffectStatistics = dspy.OutputField(
        desc="Statistics block for this estimate"
    )


class ExtractBaseline(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract the average energy consumption that the estimate refers
    to, so the effect can later be expressed as a percentage change. Use the highest available
    option in this preference order: (1) all households (treatment + control) in the baseline
    period, (2) weighted average of treatment and control in the baseline period, (3) all
    households during the treatment period, (4) weighted average of treatment and control
    during the treatment period, (5) the control group.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Candidate = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    baseline: BaselineInfo = dspy.OutputField(
        desc="Baseline consumption block for this estimate"
    )


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------

_DELIVERY_TYPES = {"Information", "Feedback", "Social Comparison"}


def _missing(value: str) -> bool:
    # Case-insensitive: "nr" and "NR" are both not reported.
    return value.strip().upper() in _NOT_REPORTED


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def _arms_key(candidate: Candidate) -> tuple[str, ...]:
    return tuple(sorted(_norm(arm) for arm in candidate.intervention_arms))


def _estimate_value(effect: Effect) -> str | None:
    if effect.coefficient_estimate is not None:
        value = effect.coefficient_estimate.coefficient
    elif effect.difference_of_means is not None:
        value = effect.difference_of_means.diff_mean
    else:
        return None
    if _missing(value):
        return None
    return _norm(value).replace("%", "").lstrip("+")


def _uncertainty(effect: Effect) -> str | None:
    if effect.coefficient_estimate is not None:
        sd = effect.coefficient_estimate.coefficient_sd
        return None if _missing(sd) else _norm(sd)
    if effect.difference_of_means is not None:
        control, treated = (
            effect.difference_of_means.control_sd,
            effect.difference_of_means.treated_sd,
        )
        if _missing(control) or _missing(treated):
            return None
        return f"{_norm(control)}|{_norm(treated)}"
    return None


def _dedupe_effects(effects: list[Effect]) -> tuple[list[Effect], list[Effect]]:
    """
    Split effects into kept and dropped copies of an estimate.

    A safety net behind select_candidates(). Two effects are copies only if the value, the
    uncertainty, the outcome and the arms all match and are reported. Matching values without
    uncertainty are only logged.
    """
    seen: dict[tuple, Effect] = {}
    values: set[tuple[str, ...]] = set()
    kept, dropped = [], []
    for effect in effects:
        value = _estimate_value(effect)
        uncertainty = _uncertainty(effect)
        if value is not None and uncertainty is not None:
            key = (
                value,
                uncertainty,
                _norm(effect.candidate.outcome),
                _arms_key(effect.candidate),
            )
            if key in seen:
                logger.info(
                    f"Dropped duplicate effect {effect.candidate.effect_number}: {key}"
                )
                dropped.append(effect)
                continue
            seen[key] = effect
        elif value is not None:
            if (value, _arms_key(effect.candidate)) in values:
                logger.warning(
                    f"Possible duplicate effect {effect.candidate.effect_number}: value {value}"
                )
            values.add((value, _arms_key(effect.candidate)))
        kept.append(effect)
    return kept, dropped


# ---------------------------------------------------------------------------
# Pipeline module
# ---------------------------------------------------------------------------


class BehaviouralInterventionExtractionPipeline(dspy.Module):
    """Gated, hierarchical extraction. See the module docstring for the query tree."""

    def __init__(self) -> None:
        super().__init__()
        self.screen = dspy.Predict(ScreenDocument)
        self.extract_study_info = dspy.Predict(ExtractStudyInfo)
        self.extract_delivery = dspy.Predict(ExtractDeliveryDetails)
        self.extract_pricing = dspy.Predict(ExtractPricingDetails)
        self.extract_motivation = dspy.Predict(ExtractMotivationDetails)
        # ChainOfThought so the reasoning behind each decision is kept
        self.classify_design = dspy.ChainOfThought(ClassifyEvaluationDesign)
        self.list_candidates = dspy.ChainOfThought(ListCandidateEstimates)
        self.extract_statistics = dspy.Predict(ExtractEffectStatistics)
        self.extract_baseline = dspy.Predict(ExtractBaseline)

    # -- L2 ----------------------------------------------------------------

    def _arm_details(self, context: str, intervention: Intervention) -> Arm_Details:
        types = set(intervention.intervention_types)
        details = Arm_Details(arm=intervention.arm)
        if types & _DELIVERY_TYPES:
            details.delivery = self.extract_delivery(
                context=context, intervention=intervention
            ).delivery
        if "Monetary Incentives" in types:
            details.pricing = self.extract_pricing(
                context=context, intervention=intervention
            ).pricing
        if "Motivation" in types:
            details.motivation = self.extract_motivation(
                context=context, intervention=intervention
            ).motivation
        return details

    # -- L3d ---------------------------------------------------------------

    def _fill_effect(self, context: str, candidate: Candidate) -> Effect:
        stats = self.extract_statistics(context=context, effect=candidate).statistics
        effect = Effect(
            candidate=candidate,
            source_quote=stats.source_quote,
            sign_convention=stats.sign_convention,
            signed_point_estimate=stats.signed_point_estimate,
            coefficient_estimate=stats.coefficient_estimate,
            difference_of_means=stats.difference_of_means,
            sample_sizes=stats.sample_sizes,
            significance=stats.significance,
        )

        # Baseline only when a percentage change must be derived (absolute units)
        if candidate.outcome_scale == "absolute":
            effect.baseline = self.extract_baseline(
                context=context, effect=candidate
            ).baseline

        return effect

    # -- forward -----------------------------------------------------------

    def forward(self, context: str) -> Study:
        # L0: screening
        screening = self.screen(context=context).screening
        if screening.exclusion_reason:
            return Study(screening=screening)

        # L1: study characteristics + treatment arms
        info = self.extract_study_info(context=context)
        study = Study(
            screening=screening,
            study_characteristics=info.study_characteristics,
            interventions=info.interventions,
        )
        if not study.interventions:
            return study

        # L2: per-arm details, only for the types each arm has
        study.arm_details = [self._arm_details(context, i) for i in study.interventions]

        # L3a: how the effects are identified
        design_step = self.classify_design(
            context=context, interventions=study.interventions
        )
        design = design_step.evaluation_design
        study.evaluation_design = design
        study.design_reasoning = design_step.reasoning
        self._apply_design(study, design)

        # L3b + L3c: list every candidate, then select the ones to code
        listing = self.list_candidates(
            context=context, interventions=study.interventions, evaluation_design=design
        )
        study.listing_reasoning = listing.reasoning
        study.candidates = select_candidates(listing.candidates)
        chosen = [record for record in study.candidates if record.selected == "Yes"]
        if not chosen:
            study.screening = Screening(
                exclusion_reason="No relevant effect sizes",
                notes_exclusion_reason="No estimates were identified after the intervention arms were found.",
            )
            return study

        # L3d + L3e: statistics and derived values for the selected estimates
        effects = [self._fill_effect(context, record.candidate) for record in chosen]
        effects, dropped = _dedupe_effects(effects)
        record_of = {id(record.candidate): record for record in study.candidates}
        for effect in dropped:
            record = record_of[id(effect.candidate)]
            record.selected = "No"
            record.decision_reason = (
                "duplicate of another selected estimate (same value and uncertainty)"
            )
        for effect in effects:
            effect.selection_reason = record_of[id(effect.candidate)].decision_reason
            effect.derived, notes = derive(effect)
            effect.notes_effect = "; ".join(notes)
        study.effects = effects
        return study

    def _apply_design(self, study: Study, design: EvaluationDesign) -> None:
        # Decided once from the design step, so it is not contradicted by a separate answer.
        if study.study_characteristics is not None:
            study.study_characteristics.randomisation_dummy = design.randomised
        for arm in study.interventions:
            arm.followup = design.post_treatment_followup
