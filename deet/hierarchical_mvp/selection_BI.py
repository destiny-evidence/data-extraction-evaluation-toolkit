"""
Deterministic selection of the estimates to code, from the candidates listed by the LLM.

The rules follow the codebook (intention-to-treat, one headline estimate per contrast, full sample
and full treatment period over subgroups and sub-periods). They do not refer to any study.
Every candidate gets a decision and a reason, so rejected estimates stay visible in the
candidates table.

A contrast is the arms, the control, the outcome and whether the estimate is a follow-up.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable

from .BehaviouralInterventionModel import Candidate, CandidateRecord

_NOT_ITT = {"ATT", "LATE"}


def select_candidates(candidates: list[Candidate]) -> list[CandidateRecord]:
    """Return one record per candidate, in the original order, with the selection decision."""
    rejected: dict[int, str] = {}
    selected: dict[int, str] = {}

    survivors = _drop_duplicates(candidates, rejected)
    averages = []
    descriptive = []
    for i in survivors:
        estimand = candidates[i].estimand_type
        if estimand == "average_effect":
            averages.append(i)
        elif estimand == "descriptive_only":
            descriptive.append(i)
        else:
            rejected[i] = f"not an average treatment effect ({estimand})"

    average_contrasts = _group_by_contrast(averages, candidates)
    for members in average_contrasts.values():
        _select_in_contrast(members, candidates, selected, rejected)

    # Descriptive means only fill a contrast that has no model-based estimate.
    for key, members in _group_by_contrast(descriptive, candidates).items():
        if key in average_contrasts:
            for i in members:
                rejected[i] = (
                    "descriptive means; a model-based estimate is selected for this contrast"
                )
        else:
            _select_in_contrast(
                members,
                candidates,
                selected,
                rejected,
                note="fallback: descriptive means, no model-based estimate for this contrast",
            )

    return [
        CandidateRecord(
            candidate=candidate,
            selected="Yes" if i in selected else "No",
            decision_reason=selected.get(i) or rejected.get(i, "not selected"),
        )
        for i, candidate in enumerate(candidates)
    ]


def _select_in_contrast(
    members: list[int],
    candidates: list[Candidate],
    selected: dict[int, str],
    rejected: dict[int, str],
    note: str = "",
) -> None:
    eligible = []
    for i in members:
        if candidates[i].analysis in _NOT_ITT:
            rejected[i] = (
                "ATT/LATE estimate; only the intention-to-treat estimate is coded"
            )
        else:
            eligible.append(i)

    # Each filter only applies when at least one candidate passes it.
    eligible = _prefer(
        eligible,
        candidates,
        rejected,
        lambda c: c.analysis != "Per-protocol or as-treated",
        "per-protocol or as-treated estimate; an intention-to-treat estimate is available",
    )
    eligible = _prefer(
        eligible,
        candidates,
        rejected,
        lambda c: c.post_treatment_selected != "Yes",
        "subgroup selected on post-treatment behaviour; an estimate for all households is available",
    )
    eligible = _prefer(
        eligible,
        candidates,
        rejected,
        lambda c: c.population == "full_sample",
        "subgroup; a pooled estimate is available for this contrast",
    )
    eligible = _prefer(
        eligible,
        candidates,
        rejected,
        lambda c: c.period != "sub_period",
        "sub-period; the full treatment period is available for this contrast",
    )
    eligible = _prefer(
        eligible,
        candidates,
        rejected,
        lambda c: c.has_uncertainty == "Yes",
        "no uncertainty reported; an estimate with uncertainty is available",
    )

    # One estimate per subgroup and sub-period. After the filters above, several groups only remain
    # when the contrast has no pooled estimate; then each group is kept (fallback).
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in eligible:
        groups[(candidates[i].population_label, candidates[i].period_label)].append(i)
    fallback = len(groups) > 1
    for group in groups.values():
        group.sort(key=lambda i: _specification_rank(candidates[i]))
        best = group[0]
        reason = note or (
            "fallback: no pooled estimate for this contrast"
            if fallback
            else "primary estimate for this contrast"
        )
        selected[best] = reason
        for i in group[1:]:
            rejected[i] = (
                f"not the selected specification for this contrast (kept {candidates[best].specification_label})"
            )


def _prefer(
    members: list[int],
    candidates: list[Candidate],
    rejected: dict[int, str],
    passes: Callable[[Candidate], bool],
    reason: str,
) -> list[int]:
    """Keep the members that pass; reject the rest. If none passes, keep all members."""
    preferred = [i for i in members if passes(candidates[i])]
    if not preferred:
        return members
    for i in members:
        if i not in preferred:
            rejected[i] = reason
    return preferred


def _specification_rank(candidate: Candidate) -> tuple:
    # Author-stated primary model first, then the most fully specified one, then the earliest.
    return (
        candidate.is_authors_preferred != "Yes",
        -candidate.n_controls_and_fixed_effects,
        _number(candidate.effect_number),
    )


def _drop_duplicates(
    candidates: list[Candidate], rejected: dict[int, str]
) -> list[int]:
    """Keep one candidate per estimate. Preferred: with uncertainty, table over text over abstract."""
    order = sorted(range(len(candidates)), key=lambda i: _preference(candidates[i]))
    kept: list[int] = []
    numbers: dict[
        int, set[str]
    ] = {}  # kept index -> effect numbers of it and its duplicates
    for i in order:
        candidate = candidates[i]
        match = next(
            (
                k
                for k in kept
                if _is_same_estimate(candidate, candidates[k], numbers[k])
            ),
            None,
        )
        if match is None:
            kept.append(i)
            numbers[i] = {_norm(candidate.effect_number)}
        else:
            rejected[i] = f"duplicate of effect {candidates[match].effect_number}"
            numbers[match].add(_norm(candidate.effect_number))
    return kept


def _is_same_estimate(
    candidate: Candidate, kept: Candidate, kept_numbers: set[str]
) -> bool:
    linked = (
        bool(candidate.same_estimate_as)
        and _norm(candidate.same_estimate_as) in kept_numbers
    )
    back_linked = bool(kept.same_estimate_as) and _norm(kept.same_estimate_as) == _norm(
        candidate.effect_number
    )
    return linked or back_linked or _label_key(candidate) == _label_key(kept)


def _preference(candidate: Candidate) -> tuple:
    location = candidate.table_or_section.lower()
    if "abstract" in location:
        rank = 2
    elif "table" in location or "figure" in location:
        rank = 0
    else:
        rank = 1
    return (candidate.has_uncertainty != "Yes", rank, _number(candidate.effect_number))


def _label_key(candidate: Candidate) -> tuple[str, str, tuple[str, ...]]:
    return (
        _norm(candidate.page),
        _norm(candidate.estimate_label),
        tuple(sorted(_norm(arm) for arm in candidate.intervention_arms)),
    )


def _contrast_key(candidate: Candidate) -> tuple[tuple[str, ...], str, str, bool]:
    return (
        tuple(sorted(_norm(arm) for arm in candidate.intervention_arms)),
        _norm(candidate.control_description),
        _norm(candidate.outcome),
        candidate.period == "post_treatment_followup",
    )


def _group_by_contrast(
    indices: list[int], candidates: list[Candidate]
) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i in indices:
        groups[_contrast_key(candidates[i])].append(i)
    return groups


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.inf
