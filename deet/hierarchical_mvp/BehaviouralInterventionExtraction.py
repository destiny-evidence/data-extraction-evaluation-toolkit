"""
DSPy signatures and gated extraction pipeline for behavioural interventions
targeting household energy use.

Same structure as the Cochrane RCT pipeline, but each level is only queried when
the level above calls for it (gating is plain Python in ``forward``, not LLM-decided):

  L0  ScreenDocument          -> stop if the document is excluded
  L1  ExtractStudyInfo        -> stop if no treatment arm is found
  L2  Extract*Details         -> only for the broad types an arm actually has
  L3a ListEffects             -> stop if no estimate is reported
  L3b Extract* (per estimate) -> only the blocks the estimate's stub requires
"""

from __future__ import annotations

import dspy

from .BehaviouralInterventionModel import (
    Arm_Details,
    BaselineInfo,
    CoefficientEstimate,
    DeliveryDetails,
    DifferenceOfMeans,
    Effect,
    Effect_Stub,
    Intervention,
    MotivationDetails,
    PricingDetails,
    SampleSizes,
    Screening,
    SignificanceInfo,
    Study,
    Study_Characteristics,
)

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


class ListEffects(dspy.Signature):
    """
    You are a systematic review assistant.

    List EVERY estimate of the effect of an intervention on household energy consumption
    that the study reports. A study may report several: one per treatment arm, population,
    time period, specification or model. Capture the intention-to-treat estimate and ignore
    ATT/LATE estimates. Only identify and classify the estimates here; their numbers are
    extracted in a later step.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    interventions: list[Intervention] = dspy.InputField(
        desc="The treatment arms already identified"
    )
    effect_stubs: list[Effect_Stub] = dspy.OutputField(
        desc="One entry per reported estimate, with its location, arm(s), design, technique and kind"
    )


class ExtractCoefficient(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract the regression coefficient, its uncertainty measure,
    the direction of the effect and the control variables used in the model.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Effect_Stub = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    coefficient_estimate: CoefficientEstimate = dspy.OutputField(
        desc="Coefficient block for this estimate"
    )


class ExtractDifferenceOfMeans(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract the control and treated group means and SDs and the
    difference of means. Capture both group means where possible.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Effect_Stub = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    difference_of_means: DifferenceOfMeans = dspy.OutputField(
        desc="Difference-of-means block for this estimate"
    )


class ExtractSampleSizes(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract the total, treatment and control sample sizes. Capture as
    much detail as possible. Several treatments may share one control group, so the treatment
    and control sizes need not add up to the total.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Effect_Stub = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    sample_sizes: SampleSizes = dspy.OutputField(
        desc="Treatment, control and total sample sizes"
    )


class ExtractTotalSampleSize(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given pre-test/post-test estimate (no control group, same group measured twice),
    extract the total sample size.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Effect_Stub = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    total_sample_size: str = dspy.OutputField(
        desc="Total number of households or observations for this estimate"
    )


class ExtractSignificance(dspy.Signature):
    """
    You are a systematic review assistant.

    For the given estimate, extract what the study reports about statistical significance:
    p-value or significance level, the test used, the test statistic and the number of tails.

    Report only information that is explicitly stated in the context. Report numbers exactly
    as they appear in the source; do not calculate or impute. If a value is not reported, use
    the string "NR".
    """

    context: str = dspy.InputField(desc=_CONTEXT_DESC)
    effect: Effect_Stub = dspy.InputField(
        desc="The estimate to extract (see its page and arm)"
    )
    significance: SignificanceInfo = dspy.OutputField(
        desc="Significance block for this estimate"
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
    effect: Effect_Stub = dspy.InputField(
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
    return value.strip() in _NOT_REPORTED


def _needs_significance(effect: Effect) -> bool:
    """Skip the significance query when the test statistic can be derived from SE/SD."""
    if effect.coefficient_estimate is not None:
        c = effect.coefficient_estimate
        return _missing(c.coefficient_sd) or c.coefficient_sd_type in {
            "NA",
            "Not provided",
        }
    if effect.difference_of_means is not None:
        d = effect.difference_of_means
        return _missing(d.control_sd) and _missing(d.treated_sd)
    return True


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
        self.list_effects = dspy.Predict(ListEffects)
        self.extract_coefficient = dspy.Predict(ExtractCoefficient)
        self.extract_difference_of_means = dspy.Predict(ExtractDifferenceOfMeans)
        self.extract_sample_sizes = dspy.Predict(ExtractSampleSizes)
        self.extract_total_sample_size = dspy.Predict(ExtractTotalSampleSize)
        self.extract_significance = dspy.Predict(ExtractSignificance)
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

    # -- L3b ---------------------------------------------------------------

    def _fill_effect(self, context: str, stub: Effect_Stub) -> Effect:
        effect = Effect(stub=stub)

        # Estimate block selected by the stub's kind
        if stub.estimate_kind == "coefficient":
            effect.coefficient_estimate = self.extract_coefficient(
                context=context, effect=stub
            ).coefficient_estimate
        else:
            effect.difference_of_means = self.extract_difference_of_means(
                context=context, effect=stub
            ).difference_of_means

        # Sample sizes: a pre/post design has no control group
        if stub.study_design == "Pre-test/post-test":
            total = self.extract_total_sample_size(
                context=context, effect=stub
            ).total_sample_size
            effect.sample_sizes = SampleSizes(total_sample_size=total)
        else:
            effect.sample_sizes = self.extract_sample_sizes(
                context=context, effect=stub
            ).sample_sizes

        # Significance only when the test statistic cannot be derived from SE/SD
        if _needs_significance(effect):
            effect.significance = self.extract_significance(
                context=context, effect=stub
            ).significance

        # Baseline only when a percentage change must be derived (absolute units)
        if stub.outcome_scale == "absolute":
            effect.baseline = self.extract_baseline(
                context=context, effect=stub
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

        # L3a: list estimates
        stubs = self.list_effects(
            context=context, interventions=study.interventions
        ).effect_stubs
        if not stubs:
            study.screening = Screening(
                exclusion_reason="No relevant effect sizes",
                notes_exclusion_reason="No estimates were identified after the intervention arms were found.",
            )
            return study

        # L3b: fill each estimate, gated by its stub
        study.effects = [self._fill_effect(context, stub) for stub in stubs]
        return study
