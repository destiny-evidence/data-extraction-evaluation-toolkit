"""Data models to help with evaluation."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from deet.data_models.base import Attribute, AttributeType

# Stable wide-CSV column order for count metrics.
COUNT_METRIC_COLUMN_ORDER: tuple[str, ...] = (
    "n_gold_instances",
    "n_good_source_instances",
    "n_good_citation_instances",
)

# Score bases that get unconditional + good/bad-source stratified columns.
_METRICS_WITH_STRATIFICATION: frozenset[str] = frozenset(
    {
        "accuracy",
        "edit_distance_match_rate",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
    }
)

_SCORE_METRIC_BASE_ORDER: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "n_labels",
    "edit_distance_match_rate",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
)

_STRATIFICATION_SUFFIXES: tuple[str, ...] = (
    "",
    "_given_good_source",
    "_given_bad_source",
)


def preferred_metric_column_names() -> list[str]:
    """
    Preferred wide-CSV metric column order.

    Derived from count names plus score bases, with stratification suffixes
    only for metrics that the evaluator stratifies by source fidelity.
    """
    columns: list[str] = list(COUNT_METRIC_COLUMN_ORDER)
    for base in _SCORE_METRIC_BASE_ORDER:
        suffixes = (
            _STRATIFICATION_SUFFIXES if base in _METRICS_WITH_STRATIFICATION else ("",)
        )
        columns.extend(f"{base}{suffix}" for suffix in suffixes)
    return columns


@dataclass(slots=True)
class AttributeMetric:
    """Base row for one metric on one attribute in an extraction run."""

    attribute: Attribute
    metric_name: str
    extraction_run_id: str


@dataclass(slots=True)
class AttributeScoreMetric(AttributeMetric):
    """A float score metric (may be ``None`` when not computable)."""

    value: float | None


@dataclass(slots=True)
class AttributeCountMetric(AttributeMetric):
    """An integer count metric (e.g. ``n_gold_instances``)."""

    value: int


class AttributeMetricsReport(BaseModel):
    """
    Per-attribute metrics block for ``metrics.json`` / wide ``metrics.csv``.

    Reuses :class:`~deet.data_models.base.AttributeType` for ``attribute_type``.
    Count keys are integers; score keys are floats (``None`` means not
    computable — omitted from JSON, blank in CSV).
    """

    attribute_id: int
    attribute_label: str
    attribute_type: AttributeType
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class RunMetricsReport(BaseModel):
    """
    Machine-readable evaluation report for one extraction run.

    Serialises to ``metrics.json`` and wide ``metrics.csv`` (one entry / row per
    attribute, metric names as keys).
    """

    extraction_run_id: str
    format_version: int = 1
    attributes: list[AttributeMetricsReport] = Field(default_factory=list)

    @classmethod
    def from_attribute_metrics(
        cls,
        *,
        extraction_run_id: str,
        calculated_metrics: list[AttributeMetric],
        format_version: int = 1,
    ) -> "RunMetricsReport":
        """
        Group row-level metric values into a run report.

        :class:`AttributeCountMetric` values go under ``counts`` as integers;
        :class:`AttributeScoreMetric` values go under ``metrics`` (including
        ``None`` for blank CSV cells). :meth:`to_json` omits ``None`` scores.

        Args:
            extraction_run_id: Run folder / run id.
            calculated_metrics: Flat metric rows from the evaluator.
            format_version: Schema version written to JSON.

        Returns:
            A :class:`RunMetricsReport` ready to serialise.

        """
        by_attribute: dict[tuple[int, str], AttributeMetricsReport] = {}
        for metric in calculated_metrics:
            attr_key = (
                metric.attribute.attribute_id,
                metric.attribute.attribute_label,
            )
            if attr_key not in by_attribute:
                by_attribute[attr_key] = AttributeMetricsReport(
                    attribute_id=metric.attribute.attribute_id,
                    attribute_label=metric.attribute.attribute_label,
                    attribute_type=metric.attribute.output_data_type,
                )
            if isinstance(metric, AttributeCountMetric):
                by_attribute[attr_key].counts[metric.metric_name] = metric.value
            elif isinstance(metric, AttributeScoreMetric):
                by_attribute[attr_key].metrics[metric.metric_name] = metric.value

        attributes = sorted(
            by_attribute.values(),
            key=lambda report: report.attribute_label,
        )
        return cls(
            extraction_run_id=extraction_run_id,
            format_version=format_version,
            attributes=attributes,
        )

    @classmethod
    def from_json(cls, filepath: Path) -> "RunMetricsReport":
        """
        Load a report from a ``metrics.json`` file.

        Args:
            filepath: Path to a JSON file written by :meth:`to_json`.

        Returns:
            Validated :class:`RunMetricsReport`.

        """
        return cls.model_validate_json(filepath.read_text(encoding="utf-8"))

    def to_json(self, filepath: Path) -> None:
        """
        Write this report to ``metrics.json``.

        ``None`` score values are omitted so inapplicable keys are absent.

        Args:
            filepath: Destination path (must end in ``.json``).

        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        for attr_payload in payload["attributes"]:
            attr_payload["metrics"] = {
                key: value
                for key, value in attr_payload["metrics"].items()
                if value is not None
            }
        filepath.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def to_csv(self, filepath: Path) -> None:
        """
        Write this report as a wide ``metrics.csv`` (one row per attribute).

        ``None`` score values become empty cells.

        Args:
            filepath: Destination path (must end in ``.csv``).

        Raises:
            ValueError: If ``filepath`` does not end in ``.csv``.

        """
        if filepath.suffix != ".csv":
            bad_filetype = "file ending must be .csv"
            raise ValueError(bad_filetype)

        base_columns = [
            "extraction_run_id",
            "attribute_id",
            "attribute_label",
            "attribute_type",
        ]
        preferred = preferred_metric_column_names()
        metric_names: set[str] = set()
        for attr_report in self.attributes:
            metric_names.update(attr_report.counts)
            metric_names.update(attr_report.metrics)

        ordered_metric_columns = [name for name in preferred if name in metric_names]
        ordered_metric_columns.extend(
            sorted(name for name in metric_names if name not in ordered_metric_columns)
        )
        fieldnames = base_columns + ordered_metric_columns

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            writer.writeheader()
            for attr_report in self.attributes:
                row: dict[str, object] = {
                    "extraction_run_id": self.extraction_run_id,
                    "attribute_id": attr_report.attribute_id,
                    "attribute_label": attr_report.attribute_label,
                    "attribute_type": attr_report.attribute_type.value,
                }
                for name in ordered_metric_columns:
                    if name in attr_report.counts:
                        row[name] = attr_report.counts[name]
                    elif name in attr_report.metrics:
                        value = attr_report.metrics[name]
                        row[name] = "" if value is None else value
                    else:
                        row[name] = ""
                writer.writerow(row)
