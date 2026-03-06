"""Data models to help with evaluation."""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from deet.data_models.base import Attribute

MetricFn = Callable[[list, list], float | np.floating | np.ndarray]


def n_labels(y_true: list[int], y_pred: list[int]) -> float:  # noqa: ARG001
    """Count the number of positive instances of the class in gold data."""
    return sum(y_true)


METRICS: dict[str, MetricFn] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
    "n_labels": n_labels,
}


class AttributeMetric(BaseModel):
    """Data structure storing a metric for an attribute for a pipeline run."""

    attribute: Attribute
    metric: str
    value: float | None

    def write_to_csv(self, filepath: Path, mode: Literal["a", "w"] = "a") -> None:
        """
        Write an evaluation metric for an attribute as a line to a csv file.

        Args:
            filepath (Path): outfile destination.
            mode (Literal["a", "w"], optional): _w_rite or _a_ppend.
            Defaults to "a" (append).

        """
        dictified = self.model_dump()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        file_exists = filepath.exists() and filepath.stat().st_size > 0
        write_header = not file_exists or mode == "w"

        with filepath.open(mode=mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dictified.keys())

            if write_header:
                writer.writeheader()

            writer.writerow(dictified)

        logger.debug(f"Wrote metric to {filepath}")
