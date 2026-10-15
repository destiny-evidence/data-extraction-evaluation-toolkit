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


def _parse_statistics(value: object, name: str) -> list[ReportedStatistic]:
    """Accept either model statistics or their serialized export representation."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        message = f"{name} must contain a list of reported statistics."
        raise TypeError(message)
    return [
        item
        if isinstance(item, ReportedStatistic)
        else ReportedStatistic.model_validate(item)
        for item in value
    ]


def expand_statistics_columns(
    fieldnames: list[str], rows: list[dict[str, object]]
) -> tuple[list[str], list[dict[str, object]]]:
    """Expand statistic lists into indexed columns without changing input rows."""
    statistic_fields = {
        f"{group}_{kind}_statistics"
        for group in ("group", "group_a", "group_b")
        for kind in ("summary", "variability")
    }
    fields = list(dict.fromkeys([*fieldnames, *(key for row in rows for key in row)]))
    selected = [name for name in fields if name in statistic_fields]
    if not selected:
        return fieldnames, rows

    parsed_rows = [
        {name: _parse_statistics(row.get(name), name) for name in selected}
        for row in rows
    ]

    reserved = {name.casefold() for name in fields}
    columns: dict[str, list[tuple[int, str, str]]] = {}
    for name in selected:
        count = max((len(row[name]) for row in parsed_rows), default=0)
        columns[name] = []
        for index in range(max(1, count)):
            for key in ReportedStatistic.model_fields:
                base = f"{name}_{index + 1}_{key}"
                column = base
                suffix = 2
                while column.casefold() in reserved:
                    column = f"{base}_{suffix}"
                    suffix += 1
                reserved.add(column.casefold())
                columns[name].append((index, key, column))

    expanded_fields = [
        column
        for name in fields
        for column in (
            [column for _, _, column in columns[name]] if name in columns else [name]
        )
    ]
    expanded_rows = []
    for row, parsed in zip(rows, parsed_rows, strict=True):
        expanded = {key: value for key, value in row.items() if key not in selected}
        for name in selected:
            statistics = parsed[name]
            for index, key, column in columns[name]:
                expanded[column] = (
                    getattr(statistics[index], key) if index < len(statistics) else ""
                )
        expanded_rows.append(expanded)
    return expanded_fields, expanded_rows


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
