"""Data models to help with evaluation."""

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from deet.data_models.base import Attribute

MetricFn = Callable[[list, list], float | np.floating | np.ndarray]

METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
}


class AttributeMetric(BaseModel):
    """Data structure storing a metric for an attribute for a pipeline run."""

    attribute: Attribute
    metric: str
    value: float | None
