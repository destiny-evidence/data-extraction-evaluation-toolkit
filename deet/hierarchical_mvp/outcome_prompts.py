"""Shared extraction instructions for arm-level RCT results."""

import dspy

BASELINE_OUTCOME_INSTRUCTIONS = (
    "Baseline measurements are outcome results, including when reported in "
    "baseline-characteristics tables. Create separate Baseline outcome objects "
    "in the appropriate outcome list; never combine baseline and follow-up "
    "measurements in one object. Put baseline numbers in the structured result "
    "or statistic fields. Where the schema provides outcome_time_point, set "
    "time_point_category to 'Baseline'; for Cochrane schemas, identify the "
    "object's baseline time point in footnotes without changing the reported "
    "outcome name. Mentioning those numbers in supplementary_info, "
    "baseline_imbalances, outcome_definition, or footnotes instead of structured "
    "results does not satisfy this requirement. "
    "Keep the schema's arm-level or pairwise grouping, with every group in an "
    "object referring to the same time point. For pairwise results, baseline "
    "values for groups A and B belong in a separate object from their follow-up "
    "values, not in the other group's fields. Extract only reported outcome "
    "measurements; do not turn every baseline demographic characteristic into "
    "an outcome. Do not infer baseline values from change scores. "
    "Do not copy follow-up sample sizes or imputation methods into baseline "
    "objects unless the source explicitly applies them to baseline."
    " For animal studies, preserve the induction/assessment intervention "
    "combination and the reported time reference (e.g. before induction or "
    "before assessment treatment); do not infer that these baselines coincide."
)

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
    "If all three arms report baseline mean (SE) and Week 8 mean change (SE), "
    "produce six continuous outcome objects: three Baseline objects containing "
    "the reported means and SEs, and three Follow-up objects containing the "
    "reported mean changes and SEs. "
    "Never combine two arms in one object or duplicate an arm for pairwise "
    "comparisons. For animal studies, retain the induction subgroup for each "
    "assessment arm. Do not invent unreported arm results or split a "
    "between-arm effect estimate into fabricated arm-level results."
    " For continuous outcomes, include at most one summary statistic and one "
    "corresponding variability statistic per outcome object. Create a new outcome "
    "object for each additional statistic set, including endpoint versus change "
    "scores, mean versus median, or multiple variability measures. Repeat a summary "
    "statistic only when the source associates it with the additional variability "
    "measure. Never pair unrelated statistics or discard reported measures."
)


def baseline_outcome_signature(
    signature: type[dspy.Signature],
) -> type[dspy.Signature]:
    """Require structured baseline objects without changing the grouping schema."""
    return signature.with_instructions(
        signature.instructions + "\n\n" + BASELINE_OUTCOME_INSTRUCTIONS
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
