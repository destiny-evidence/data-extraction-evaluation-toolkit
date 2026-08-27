"""Data models to help with evaluation."""

import csv
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from deet.data_models.base import Attribute


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
