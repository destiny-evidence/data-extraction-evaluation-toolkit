"""Metric functions and registries for gold-vs-LLM evaluation."""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import (  # type:ignore[import-untyped]
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
)

from deet.data_models.base import AttributeType
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


def edit_distance_match_rate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    threshold: float = DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD,
) -> float:
    """
    Fraction of pairs whose normalised Levenshtein similarity meets a threshold.

    Values are normalised with :func:`normalize_string_for_match` before
    comparison. Missing predictions are not dropped: any ``None`` in
    ``y_pred`` raises, matching binary-metric behaviour (the evaluator then
    records ``value=None`` for the metric).

    Args:
        y_true: Gold-standard values.
        y_pred: Predicted values.
        threshold: Minimum normalised similarity in ``[0, 1]`` to count as a
            match. Defaults to ``0.90``.

    Returns:
        Match rate in ``[0.0, 1.0]``. Returns ``0.0`` when both lists are empty.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths.
        TypeError: If any prediction is ``None``.

    """
    if len(y_true) != len(y_pred):
        msg = (
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )
        raise ValueError(msg)
    if any(pred_val is None for pred_val in y_pred):
        msg = "edit_distance_match_rate does not accept None predictions"
        raise TypeError(msg)
    if not y_true:
        return 0.0

    matches = 0
    for true_val, pred_val in zip(y_true, y_pred, strict=True):
        true_norm = normalize_string_for_match(str(true_val))
        pred_norm = normalize_string_for_match(str(pred_val))
        similarity = Levenshtein.normalized_similarity(true_norm, pred_norm)
        if similarity >= threshold:
            matches += 1
    return matches / len(y_true)


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

# Exact match plus sklearn regression metrics on numeric values.
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
