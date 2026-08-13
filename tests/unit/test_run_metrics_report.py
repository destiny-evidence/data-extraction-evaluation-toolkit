"""Unit tests for the machine-readable metrics.json report models."""

from __future__ import annotations

import json
from pathlib import Path

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.evaluation import AttributeMetric, RunMetricsReport

_DUMMY_METRICS_JSON = (
    Path(__file__).resolve().parents[2]
    / "misc"
    / "smoking_mapfile_required"
    / "data-extraction-experiments"
    / "2026-08-03_10-13-40_"
    / "metrics_dummy.json"
)


def test_run_metrics_report_round_trips_dummy_json() -> None:
    """The dummy metrics.json file validates and serialises back to the same payload."""
    original = json.loads(_DUMMY_METRICS_JSON.read_text(encoding="utf-8"))
    report = RunMetricsReport.from_json(_DUMMY_METRICS_JSON)
    assert report.model_dump(mode="json") == original


def test_run_metrics_report_from_attribute_metrics_omits_nulls() -> None:
    """None metric values are omitted; counts are integers."""
    attribute = Attribute(
        attribute_id=1,
        attribute_label="Outcome label",
        output_data_type=AttributeType.STRING,
    )
    calculated_metrics = [
        AttributeMetric(
            attribute=attribute,
            metric_name="n_gold_instances",
            value=10.0,
            extraction_run_id="run_1",
        ),
        AttributeMetric(
            attribute=attribute,
            metric_name="accuracy",
            value=0.6,
            extraction_run_id="run_1",
        ),
        AttributeMetric(
            attribute=attribute,
            metric_name="accuracy_given_good_source",
            value=None,
            extraction_run_id="run_1",
        ),
    ]
    report = RunMetricsReport.from_attribute_metrics(
        extraction_run_id="run_1",
        calculated_metrics=calculated_metrics,
    )
    dumped = report.model_dump(mode="json")
    assert dumped["format_version"] == 1
    assert dumped["attributes"][0]["attribute_type"] == "string"
    assert dumped["attributes"][0]["counts"] == {"n_gold_instances": 10}
    assert dumped["attributes"][0]["metrics"] == {"accuracy": 0.6}
    assert "accuracy_given_good_source" not in dumped["attributes"][0]["metrics"]
