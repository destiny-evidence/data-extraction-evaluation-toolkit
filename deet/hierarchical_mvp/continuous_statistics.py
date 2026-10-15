"""Reported continuous statistics shared by all RCT schemas."""

import json

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo


class ReportedStatistic(BaseModel):
    """One named statistic with its reported value and interpretation context."""

    measure: str = Field(
        description=(
            "Name of the reported measure. Suggested summary measures: Mean, Median, "
            "Geometric mean, Mean change, Other. Suggested variability measures: "
            "Standard deviation, Standard error, Interquartile range, Range, "
            "Confidence interval, Variance, Other. Use the source's specific name "
            "for an unlisted measure; do not relabel one measure as another."
        )
    )
    value: str = Field(
        description=(
            "Value exactly as reported, including signs, inequalities, or both bounds "
            "of an interval (e.g. '8-17'). Do not calculate or convert statistics."
        )
    )
    basis: str = Field(
        default="NR",
        description=(
            "What the statistic describes: Endpoint, Change from baseline, or NR "
            "if not reported. Never assign a between-arm estimate to one arm."
        ),
    )
    details: str = Field(
        default="",
        description=(
            "Reported context needed to interpret this statistic: e.g. '95% CI for "
            "the mean', 'Q1-Q3', adjusted/unadjusted, or the specific Other measure. "
            "Associate variability with its corresponding summary measure here."
        ),
    )


def summary_statistics_field(arm: str) -> FieldInfo:
    """Describe a list of arm-level location or change statistics."""
    return Field(
        description=(
            f"All reported summary statistics for {arm}, such as mean, median, "
            "geometric mean, or within-arm mean change. Keep multiple statistics "
            "in this list without duplicating the outcome. Distinguish endpoint "
            "and change scores using basis. Never put a between-arm mean difference "
            "here or infer arm results from it; retain comparisons in effect_estimates "
            "or footnotes. Use [] when no arm-level summary statistic is reported."
        )
    )


def variability_statistics_field(arm: str) -> FieldInfo:
    """Describe a list of reported dispersion or uncertainty statistics."""
    return Field(
        description=(
            f"All reported variability/uncertainty statistics for {arm}: Standard "
            "deviation, Standard error, Interquartile range, Range, Confidence "
            "interval, Variance, or Other. Keep every reported measure, interval "
            "bounds and confidence level; identify the related summary statistic "
            "in details and endpoint/change in basis. Do not convert SE to SD, "
            "IQR to SD, or infer missing statistics. Use [] if none are reported."
        )
    )


def statistics_json(statistics: list[ReportedStatistic]) -> str:
    """Keep named statistics lossless in a single CSV/XLSX cell."""
    return json.dumps([item.model_dump() for item in statistics], ensure_ascii=False)


def revman_statistics(
    summary: list[ReportedStatistic],
    variability: list[ReportedStatistic],
) -> dict[str, str]:
    """Project only unambiguous endpoint statistics into fixed RevMan columns."""
    columns = {}
    for column, measure, statistics in (
        ("Mean", "Mean", summary),
        ("SD", "Standard deviation", variability),
        ("SE", "Standard error", variability),
        ("Variance", "Variance", variability),
    ):
        matches = [
            item.value
            for item in statistics
            if item.measure.casefold() == measure.casefold()
            and item.basis.casefold() == "endpoint"
            and not item.details
        ]
        columns[column] = matches[0] if len(matches) == 1 else ""
    return {
        **columns,
        "group_summary_statistics": statistics_json(summary),
        "group_variability_statistics": statistics_json(variability),
    }
