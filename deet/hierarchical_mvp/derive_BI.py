"""
Deterministic derivations for the behavioural-intervention effects (no LLM involved).

The LLM reports only what the paper states. The values below are computed from the reported
fields and written to ``derived_*`` columns next to the reported ones. A derived value is empty
when an input is missing or not numeric. Where a derived value disagrees with the reported value,
the disagreement is returned as a note and nothing is overwritten.
"""

from __future__ import annotations

import math
import re

from .BehaviouralInterventionModel import DERIVED_FIELDS, Effect, SignificanceInfo

_MISSING = {"", "NR", "NA"}
# Star legend entries such as "*** p<0.01": number of stars and the significance level.
_LEGEND_ENTRY = re.compile(r"(\*+)\s*p\s*<\s*(\d*\.?\d+)")


def derive(effect: Effect) -> tuple[dict[str, str], list[str]]:
    """Return the derived values (keys are DERIVED_FIELDS) and the mismatch notes."""
    signed = to_float(effect.signed_point_estimate)
    change = (
        None if signed is None else consumption_change(signed, effect.sign_convention)
    )
    derived = dict.fromkeys(DERIVED_FIELDS, "")
    if change is not None:
        derived["derived_effect_direction"] = effect_direction(change)
        derived["derived_percentage_change"] = percentage_change(
            change, effect.candidate.outcome_scale, _baseline(effect)
        )
    derived["derived_diff_mean"] = _diff_mean(effect)
    derived["derived_test_statistic"] = _test_statistic(effect, signed)
    derived["derived_total_sample_size"] = _total_sample_size(effect)
    reporting, p_bound = significance_bound(effect.significance)
    derived["derived_significance_reporting"] = reporting
    derived["derived_p_bound"] = p_bound
    return derived, _mismatches(effect, derived)


def to_float(value: str) -> float | None:
    text = value.strip().replace("−", "-").replace(",", "").replace("%", "")
    if text.upper() in _MISSING:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def consumption_change(signed: float, sign_convention: str) -> float:
    """
    Change in consumption, negative if consumption fell.

    For a 'reduction (positive = saving)' column the printed sign is flipped.
    """
    return -signed if sign_convention.startswith("reduction") else signed


def effect_direction(change: float) -> str:
    if change == 0:
        return "No change"
    return "Increase" if change > 0 else "Decrease"


def percentage_change(change: float, outcome_scale: str, baseline: float | None) -> str:
    """
    Percentage change in consumption.

    'percent' outcomes are already a percentage. 'log' outcomes use exp(b) - 1. 'absolute'
    outcomes divide the change by the baseline consumption.
    """
    match outcome_scale:
        case "percent":
            value = change
        case "log":
            value = (math.exp(change) - 1) * 100
        case _:
            if baseline is None or baseline == 0:
                return ""
            value = change / baseline * 100
    return fmt(value)


def significance_bound(significance: SignificanceInfo | None) -> tuple[str, str]:
    """Return (significance_reporting, p_bound) from a reported p-value or from the stars and legend."""
    if significance is None:
        return "", ""
    p_value = to_float(significance.p_value)
    if p_value is not None:
        return "actual", fmt(p_value)
    levels = {
        len(m.group(1)): float(m.group(2))
        for m in _LEGEND_ENTRY.finditer(significance.star_legend)
    }
    if not levels:
        return "", ""
    stars = significance.stars.strip()
    if not stars:
        # No star: not significant at the least strict level in the legend.
        return "lower bound", fmt(max(levels.values()))
    if len(stars) not in levels:
        return "", ""
    return "upper bound", fmt(levels[len(stars)])


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    return format(round(value, 6), "f").rstrip("0").rstrip(".")


def _baseline(effect: Effect) -> float | None:
    if effect.baseline is None:
        return None
    return to_float(effect.baseline.baseline_consumption)


def _diff_mean(effect: Effect) -> str:
    block = effect.difference_of_means
    if block is None:
        return ""
    treated, control = to_float(block.treated_mean), to_float(block.control_mean)
    if treated is None or control is None:
        return ""
    return fmt(treated - control)


def _test_statistic(effect: Effect, signed: float | None) -> str:
    block = effect.coefficient_estimate
    if block is None or signed is None:
        return ""
    se = to_float(block.coefficient_sd)
    if se is None or se <= 0 or "error" not in block.coefficient_sd_type.lower():
        return ""
    return fmt(abs(signed) / se)


def _total_sample_size(effect: Effect) -> str:
    sizes = effect.sample_sizes
    if sizes is None:
        return ""
    treatment, control = (
        to_float(sizes.treatment_sample_size),
        to_float(sizes.control_sample_size),
    )
    if treatment is None or control is None:
        return ""
    return fmt(treatment + control)


def _mismatches(effect: Effect, derived: dict[str, str]) -> list[str]:
    reported_direction = ""
    if effect.coefficient_estimate is not None:
        reported_direction = effect.coefficient_estimate.effect_direction
    elif effect.difference_of_means is not None:
        reported_direction = effect.difference_of_means.effect_direction
    reported_diff = (
        effect.difference_of_means.diff_mean if effect.difference_of_means else ""
    )
    reported_total = (
        effect.sample_sizes.total_sample_size if effect.sample_sizes else ""
    )
    reported_reporting = (
        effect.significance.significance_reporting if effect.significance else ""
    )

    checks = [
        ("effect_direction", reported_direction, derived["derived_effect_direction"]),
        ("diff_mean", reported_diff, derived["derived_diff_mean"]),
        ("total_sample_size", reported_total, derived["derived_total_sample_size"]),
        (
            "significance_reporting",
            reported_reporting,
            derived["derived_significance_reporting"],
        ),
    ]
    return [
        f"check {name}: reported {reported} vs derived {value}"
        for name, reported, value in checks
        if _differs(reported, value)
    ]


def _differs(reported: str, derived: str) -> bool:
    if reported.strip().upper() in _MISSING or not derived:
        return False
    reported_number, derived_number = to_float(reported), to_float(derived)
    if reported_number is not None and derived_number is not None:
        return not math.isclose(
            reported_number, derived_number, rel_tol=0.02, abs_tol=0.01
        )
    return reported.strip().lower() != derived.lower()
