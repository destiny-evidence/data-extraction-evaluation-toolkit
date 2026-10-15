"""Shared extraction instructions for arm-level RCT results."""

import dspy

CONTINUOUS_STATISTICS_INSTRUCTIONS = (
    "Extract every reported arm-level summary and variability measure as a "
    "named statistic, including means, medians, within-arm changes, SD, SE, IQR "
    "and confidence intervals. Preserve all reported statistics rather than "
    "choosing one. Keep endpoint and change-from-baseline statistics distinct "
    "using basis; identify related summary measures and confidence levels in "
    "details. Do not convert statistics or treat between-arm mean differences "
    "as arm-level averages. Preserve comparisons in effect_estimates or footnotes. "
    "For unreported statistic lists return [], not 'NR' or fabricated entries."
)

SINGLE_ARM_OUTCOME_INSTRUCTIONS = (
    "Extract exactly one intervention arm (one group) per outcome object. "
    "Create a separate object for every reported outcome, time point, and arm, "
    "including control arms and baseline measurements. "
    "Set group_label to the corresponding identified intervention name, and put "
    "only that arm's results in the group fields. "
    "For example, MADRS results reported for all three arms at one time point "
    "must produce three continuous outcome objects, one per arm. "
    "Never combine two arms in one object or duplicate an arm for pairwise "
    "comparisons. For animal studies, retain the induction subgroup for each "
    "assessment arm. Do not invent unreported arm results or split a "
    "between-arm effect estimate into fabricated arm-level results."
)


def single_arm_outcome_signature(
    signature: type[dspy.Signature],
) -> type[dspy.Signature]:
    """Add the common arm-level contract without discarding subtype instructions."""
    return signature.with_instructions(
        signature.instructions + "\n\n" + SINGLE_ARM_OUTCOME_INSTRUCTIONS
    )


def continuous_outcome_signature(
    signature: type[dspy.Signature],
) -> type[dspy.Signature]:
    """Apply the same named-statistic contract to static and custom extraction."""
    return signature.with_instructions(
        signature.instructions + "\n\n" + CONTINUOUS_STATISTICS_INSTRUCTIONS
    )
