"""Data models to help with evaluation."""

import csv
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from deet.data_models.base import Attribute, AttributeType

COUNT_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "n_gold_instances",
        "n_good_source_instances",
        "n_good_citation_instances",
    }
)


class AttributeMetric(BaseModel):
    """Data structure storing a metric for an attribute for a data extraction run."""

    attribute: Attribute
    metric_name: str
    value: float | None
    extraction_run_id: str

    def dictify(self) -> dict:
        """
        Return a dictionary representation, unpacking the attribute into ID
            and label.
        """
        return {
            "attribute_id": self.attribute.attribute_id,
            "attribute_label": self.attribute.attribute_label,
            "value": self.value,
            "extraction_run_id": self.extraction_run_id,
            "metric_name": self.metric_name,
        }

    def save_to_csv(self, filepath: Path, mode: Literal["a", "w"] = "a") -> None:
        """
        Write an evaluation metric for an attribute as a line to a csv file.

        Args:
            filepath (Path): outfile destination.
            mode (Literal["a", "w"], optional): _w_rite or _a_ppend.
            Defaults to "a" (append).

        """
        dictified = self.dictify()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        write_header = not file_exists or mode == "w"

        with filepath.open(mode=mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dictified.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(dictified)

        logger.debug(f"Wrote metric to {filepath}")


class AttributeMetricsReport(BaseModel):
    """
    Per-attribute metrics block written to ``metrics.json``.

    Reuses :class:`~deet.data_models.base.AttributeType` for ``attribute_type``.
    Count keys are integers; score keys are floats. Inapplicable keys are omitted.
    """

    attribute_id: int
    attribute_label: str
    attribute_type: AttributeType
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class RunMetricsReport(BaseModel):
    """
    Machine-readable evaluation report for one extraction run.

    Serialises to ``metrics.json`` with the same shape as the human-readable
    wide ``metrics.csv`` (one entry per attribute, metric names as keys).
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
        Group row-level :class:`AttributeMetric` values into a run report.

        Count metrics (``n_gold_instances`` and source/citation counts) are
        stored as integers under ``counts``; remaining scores go under
        ``metrics``. ``None`` values are omitted so inapplicable keys are
        absent from the JSON.

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
            if metric.value is None:
                continue
            if metric.metric_name in COUNT_METRIC_NAMES:
                by_attribute[attr_key].counts[metric.metric_name] = int(metric.value)
            else:
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

        Args:
            filepath: Destination path (must end in ``.json``).

        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
