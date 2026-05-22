"""Data models to help with evaluation."""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import (  # type:ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from deet.data_models.base import Attribute, AttributeType

MetricFunction = Callable[[list, list], float | np.floating | np.ndarray]


def check_metric_returns_float(metric: MetricFunction) -> bool:
    """Check whether a metric returns a scalar."""
    y_true = [1, 0, 0, 1]
    y_pred = [1, 0, 0, 0]
    result = metric(y_true, y_pred)
    return isinstance(result, float)


def n_labels(y_true: list[int], y_pred: list[int]) -> float:  # noqa: ARG001
    """Count the number of positive instances of the class in gold data."""
    return sum(y_true)


BINARY_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
    "n_labels": n_labels,
}

# Per-value exact match; appropriate for categorical string extraction.
STRING_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
}

# Discrete integer extraction; per-value match via sklearn accuracy.
INTEGER_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
}

# Regression-style metrics (e.g. MAE) are not wired yet.
FLOAT_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
}

# Structured values need dedicated metrics (set overlap, tree edit distance, etc.).
LIST_METRICS: dict[str, MetricFunction] = {}
DICT_METRICS: dict[str, MetricFunction] = {}

# Keep METRICS as the default (boolean) set for backward compatibility
METRICS: dict[str, MetricFunction] = BINARY_METRICS

METRICS_BY_ATTRIBUTE_TYPE: dict[AttributeType, dict[str, MetricFunction]] = {
    AttributeType.BOOL: BINARY_METRICS,
    AttributeType.STRING: STRING_METRICS,
    AttributeType.INTEGER: INTEGER_METRICS,
    AttributeType.FLOAT: FLOAT_METRICS,
    AttributeType.LIST: LIST_METRICS,
    AttributeType.DICT: DICT_METRICS,
}


def get_metrics_for_attribute_type(
    attribute_type: AttributeType,
) -> dict[str, MetricFunction]:
    """
    Return the metric set registered for the given attribute data type.

    Some types map to an empty dict when no suitable default metrics are
    implemented yet (float, list, dict); callers may still merge in custom metrics.
    """
    return METRICS_BY_ATTRIBUTE_TYPE[attribute_type]


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
