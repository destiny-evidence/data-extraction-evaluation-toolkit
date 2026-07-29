"""Data models to help with evaluation."""

import csv
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import (  # type:ignore[import-untyped]
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from deet.data_models.base import Attribute, AttributeType
from deet.utils.text_normalisation import normalize_string_for_match

MetricFunction = Callable[[list, list], float | np.floating | np.ndarray]

DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD: float = 0.90


class EvaluationMetricSettings(BaseModel):
    """Configurable thresholds for extraction evaluation metrics."""

    edit_distance_match_threshold: float = Field(
        default=DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum normalised Levenshtein similarity (0-1) for a string pair "
            "to count as a match in edit_distance_match_rate."
        ),
    )


def check_metric_returns_float(metric: MetricFunction) -> bool:
    """Check whether a metric returns a scalar."""
    y_true = [1, 0, 0, 1]
    y_pred = [1, 0, 0, 0]
    result = metric(y_true, y_pred)
    return isinstance(result, float)


def n_labels(y_true: list[int], y_pred: list[int]) -> float:  # noqa: ARG001
    """Count the number of positive instances of the class in gold data."""
    return sum(y_true)


def filter_valid_pairs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[list[Any], list[Any]]:
    """
    Drop pairs where the prediction is ``None``.

    Used by regression-style and near-match metrics so missing LLM outputs do
    not poison the score. Gold and prediction lists stay aligned.

    Args:
        y_true: Gold-standard values.
        y_pred: Predicted values (may contain ``None``).

    Returns:
        Filtered ``(y_true, y_pred)`` lists with ``None`` predictions removed.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.

    """
    if len(y_true) != len(y_pred):
        msg = (
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )
        raise ValueError(msg)

    filtered_true: list[Any] = []
    filtered_pred: list[Any] = []
    for true_val, pred_val in zip(y_true, y_pred, strict=True):
        if pred_val is None:
            continue
        filtered_true.append(true_val)
        filtered_pred.append(pred_val)
    return filtered_true, filtered_pred


def edit_distance_match_rate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    threshold: float = DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD,
) -> float:
    """
    Fraction of pairs whose normalised Levenshtein similarity meets a threshold.

    Predictions that are ``None`` are dropped. Surviving values are normalised
    with :func:`normalize_string_for_match` before comparison.

    Args:
        y_true: Gold-standard values.
        y_pred: Predicted values.
        threshold: Minimum normalised similarity in ``[0, 1]`` to count as a
            match. Defaults to ``0.90``.

    Returns:
        Match rate in ``[0.0, 1.0]``. Returns ``0.0`` when no valid pairs remain.

    """
    filtered_true, filtered_pred = filter_valid_pairs(y_true, y_pred)
    if not filtered_true:
        logger.debug("edit_distance_match_rate: no valid pairs after filtering")
        return 0.0

    matches = 0
    for true_val, pred_val in zip(filtered_true, filtered_pred, strict=True):
        true_norm = normalize_string_for_match(str(true_val))
        pred_norm = normalize_string_for_match(str(pred_val))
        similarity = Levenshtein.normalized_similarity(true_norm, pred_norm)
        if similarity >= threshold:
            matches += 1
    return matches / len(filtered_true)


def _coerce_numeric_pairs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[list[float], list[float]]:
    """
    Filter ``None`` predictions and coerce remaining values to ``float``.

    Args:
        y_true: Gold-standard values.
        y_pred: Predicted values.

    Returns:
        Parallel lists of coerced floats.

    Raises:
        ValueError: If no valid pairs remain after filtering.
        TypeError: If a value cannot be coerced to float.

    """
    filtered_true, filtered_pred = filter_valid_pairs(y_true, y_pred)
    if not filtered_true:
        msg = "No valid numeric pairs after filtering None predictions"
        raise ValueError(msg)

    coerced_true: list[float] = []
    coerced_pred: list[float] = []
    for true_val, pred_val in zip(filtered_true, filtered_pred, strict=True):
        coerced_true.append(float(true_val))
        coerced_pred.append(float(pred_val))
    return coerced_true, coerced_pred


def mean_absolute_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """
    Mean absolute error on coerced numeric pairs.

    Drops ``None`` predictions, then coerces remaining values to ``float``.

    Args:
        y_true: Gold-standard numeric values.
        y_pred: Predicted numeric values.

    Returns:
        Mean of absolute differences.

    Raises:
        ValueError: If no valid pairs remain after filtering.
        TypeError: If a value cannot be coerced to float.

    """
    coerced_true, coerced_pred = _coerce_numeric_pairs(y_true, y_pred)
    absolute_errors = [
        abs(true_val - pred_val)
        for true_val, pred_val in zip(coerced_true, coerced_pred, strict=True)
    ]
    return sum(absolute_errors) / len(absolute_errors)


def mean_absolute_percentage_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """
    Mean absolute percentage error on coerced numeric pairs.

    Drops ``None`` predictions and pairs where the gold value is ``0`` (logs a
    warning for each skipped zero-gold pair). Percentage error is
    ``|pred - true| / |true|`` (not multiplied by 100).

    Args:
        y_true: Gold-standard numeric values.
        y_pred: Predicted numeric values.

    Returns:
        Mean absolute percentage error as a fraction.

    Raises:
        ValueError: If no pairs remain after filtering ``None`` predictions and
            zero-gold values.
        TypeError: If a value cannot be coerced to float.

    """
    coerced_true, coerced_pred = _coerce_numeric_pairs(y_true, y_pred)

    percentage_errors: list[float] = []
    skipped_zero_gold = 0
    for true_val, pred_val in zip(coerced_true, coerced_pred, strict=True):
        if true_val == 0.0:
            skipped_zero_gold += 1
            continue
        percentage_errors.append(abs(pred_val - true_val) / abs(true_val))

    if skipped_zero_gold:
        logger.warning(
            f"mean_absolute_percentage_error: skipped {skipped_zero_gold} "
            "pair(s) with zero gold value"
        )

    if not percentage_errors:
        msg = (
            "No valid pairs for MAPE after skipping zero gold values "
            f"(skipped={skipped_zero_gold})"
        )
        raise ValueError(msg)

    return sum(percentage_errors) / len(percentage_errors)


BINARY_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
    "n_labels": n_labels,
}

# Per-value exact match plus near-match via normalised Levenshtein similarity.
STRING_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "edit_distance_match_rate": partial(
        edit_distance_match_rate,
        threshold=DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD,
    ),
}

# Exact match plus magnitude-of-error metrics on coerced numeric values.
INTEGER_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "mean_absolute_error": mean_absolute_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
}

FLOAT_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "mean_absolute_error": mean_absolute_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
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
    settings: EvaluationMetricSettings | None = None,
) -> dict[str, MetricFunction]:
    """
    Return the metric set registered for the given attribute data type.

    For STRING attributes, ``edit_distance_match_rate`` is rebuilt from
    ``settings.edit_distance_match_threshold`` (defaults to 0.90).

    Some types map to an empty dict when no suitable default metrics are
    implemented yet (list, dict); callers may still merge in custom metrics.

    Args:
        attribute_type: Attribute output data type.
        settings: Optional metric settings; defaults used when ``None``.

    Returns:
        Mapping of metric name to callable.

    """
    resolved_settings = settings or EvaluationMetricSettings()
    metrics = dict(METRICS_BY_ATTRIBUTE_TYPE[attribute_type])
    if attribute_type == AttributeType.STRING:
        metrics["edit_distance_match_rate"] = partial(
            edit_distance_match_rate,
            threshold=resolved_settings.edit_distance_match_threshold,
        )
    return metrics


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
