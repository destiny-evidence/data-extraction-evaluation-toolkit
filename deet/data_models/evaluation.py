"""Data models to help with evaluation."""

from abc import ABC, abstractmethod
from enum import StrEnum, auto
from uuid import UUID

from pydantic import BaseModel

from deet.data_models.base import Attribute


class EvaluationMetric(ABC):
    """Base class for an evaluation metric."""

    @abstractmethod
    def calculate(self, y_true: list, y_pred: list) -> float:
        """Calculate the evaluation metric."""
        ...


class Accuracy(EvaluationMetric):
    """Accuracy: the proportion of decisions that are correct."""

    def calculate(self, y_true: list, y_pred: list) -> float:
        """Calculate accuracy."""
        return sum(t == p for t, p in zip(y_true, y_pred, strict=False)) / len(y_true)


class Recall(EvaluationMetric):
    """Recall: the proportion of relevant items that are correctly identified."""

    def calculate(self, y_true: list, y_pred: list) -> float:
        """Calculate recall."""
        tp = sum(t == p == 1 for t, p in zip(y_true, y_pred, strict=False))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=False))
        return tp / (tp + fn) if (tp + fn) else 0.0


class MetricType(StrEnum):
    """Enum of supported metrics."""

    ACCURACY = auto()
    RECALL = auto()

    def _metric(self) -> type[EvaluationMetric]:
        """Get the metric class from the enum element."""
        mapping: dict[MetricType, type[EvaluationMetric]] = {
            MetricType.ACCURACY: Accuracy,
            MetricType.RECALL: Recall,
        }
        return mapping[self]

    def calculate(self, y_true: list, y_pred: list) -> float:
        """Calculate the metric of the metric class."""
        metric = self._metric()()
        return metric.calculate(y_true, y_pred)


class AttributeMetric(BaseModel):
    """Data structure storing a metric for an attribute for a pipeline run."""

    attribute: Attribute
    pipeline_run_id: UUID
    metric: MetricType
    value: float | None
