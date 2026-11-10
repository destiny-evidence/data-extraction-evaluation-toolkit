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
from deet.evaluators import embeddings as embedding_service
from deet.utils.text_normalisation import normalize_string_for_match

MetricFunction = Callable[[list, list], float | np.floating | np.ndarray]

DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD: float = 0.90
DEFAULT_SEMANTIC_MATCH_THRESHOLD: float = 0.85


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
    enable_semantic_metrics: bool = Field(
        default=False,
        description=(
            "When True, include semantic_match_rate for STRING attributes. "
            "Requires the optional semantic extra (fastembed). Off by default."
        ),
    )
    semantic_match_threshold: float = Field(
        default=DEFAULT_SEMANTIC_MATCH_THRESHOLD,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum embedding cosine similarity (0-1) for a string pair "
            "to count as a match in semantic_match_rate."
        ),
    )
    embedding_model_name: str = Field(
        default=embedding_service.DEFAULT_EMBEDDING_MODEL_NAME,
        description="fastembed model id used for semantic_match_rate.",
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
        Match rate in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths, or
            either list is empty (mirroring sklearn empty-input behaviour).
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
        msg = "edit_distance_match_rate requires non-empty y_true and y_pred"
        raise ValueError(msg)

    matches = 0
    for true_val, pred_val in zip(y_true, y_pred, strict=True):
        true_norm = normalize_string_for_match(str(true_val))
        pred_norm = normalize_string_for_match(str(pred_val))
        similarity = Levenshtein.normalized_similarity(true_norm, pred_norm)
        if similarity >= threshold:
            matches += 1
    return matches / len(y_true)


def semantic_match_rate(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    threshold: float = DEFAULT_SEMANTIC_MATCH_THRESHOLD,
    model_name: str = embedding_service.DEFAULT_EMBEDDING_MODEL_NAME,
) -> float:
    """
    Fraction of pairs whose embedding cosine similarity meets a threshold.

    Values are normalised with :func:`normalize_string_for_match` before
    embedding. Unique strings are encoded in one batch. Missing predictions
    are not dropped: any ``None`` in ``y_pred`` raises, matching
    :func:`edit_distance_match_rate`.

    Requires the optional ``semantic`` extra (``fastembed``). This metric is
    for string paraphrases. Numeric source fidelity stays on numeric equality.

    Args:
        y_true: Gold-standard values.
        y_pred: Predicted values.
        threshold: Minimum cosine similarity in ``[0, 1]`` to count as a
            match. Defaults to ``0.85``.
        model_name: fastembed model id.

    Returns:
        Match rate in ``[0.0, 1.0]``.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths, or
            either list is empty (mirroring sklearn empty-input behaviour).
        TypeError: If any prediction is ``None``.
        ImportError: If the ``semantic`` extra is not installed.

    """
    if len(y_true) != len(y_pred):
        msg = (
            f"y_true and y_pred must have the same length "
            f"(got {len(y_true)} and {len(y_pred)})"
        )
        raise ValueError(msg)
    if any(pred_val is None for pred_val in y_pred):
        msg = "semantic_match_rate does not accept None predictions"
        raise TypeError(msg)
    if not y_true:
        msg = "semantic_match_rate requires non-empty y_true and y_pred"
        raise ValueError(msg)

    true_norm = [normalize_string_for_match(str(true_val)) for true_val in y_true]
    pred_norm = [normalize_string_for_match(str(pred_val)) for pred_val in y_pred]
    unique_texts = list(dict.fromkeys([*true_norm, *pred_norm]))
    embeddings = embedding_service.encode_texts(unique_texts, model_name=model_name)
    index = {text: row for row, text in enumerate(unique_texts)}
    true_vectors = embeddings[[index[text] for text in true_norm]]
    pred_vectors = embeddings[[index[text] for text in pred_norm]]
    similarities = np.sum(true_vectors * pred_vectors, axis=1)
    similarities = np.clip(similarities, -1.0, 1.0)
    return float(np.mean(similarities >= threshold))


BINARY_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
    "n_labels": n_labels,
}

# Per-value exact match plus near-match via Levenshtein and, when enabled,
# embeddings. semantic_match_rate is listed so metrics.csv column order stays
# stable; get_metrics_for_attribute_type includes it only when
# enable_semantic_metrics is set.
STRING_METRICS: dict[str, MetricFunction] = {
    "accuracy": accuracy_score,
    "edit_distance_match_rate": partial(
        edit_distance_match_rate,
        threshold=DEFAULT_EDIT_DISTANCE_MATCH_THRESHOLD,
    ),
    "semantic_match_rate": partial(
        semantic_match_rate,
        threshold=DEFAULT_SEMANTIC_MATCH_THRESHOLD,
        model_name=embedding_service.DEFAULT_EMBEDDING_MODEL_NAME,
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
    ``semantic_match_rate`` is included only when
    ``settings.enable_semantic_metrics`` is True (the default is False).

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
        if resolved_settings.enable_semantic_metrics:
            metrics["semantic_match_rate"] = partial(
                semantic_match_rate,
                threshold=resolved_settings.semantic_match_threshold,
                model_name=resolved_settings.embedding_model_name,
            )
        else:
            metrics.pop("semantic_match_rate", None)
    return metrics
