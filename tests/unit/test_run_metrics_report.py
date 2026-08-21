"""Unit tests for the machine-readable metrics.json report models."""

import csv

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.evaluation import (
    AttributeCountMetric,
    AttributeScoreMetric,
    RunMetricsReport,
)


def test_run_metrics_report_round_trips_json(tmp_path) -> None:
    """A report written to JSON loads back with the same payload."""
    attribute = Attribute(
        attribute_id=1,
        attribute_label="Outcome label",
        output_data_type=AttributeType.STRING,
    )
    report = RunMetricsReport.from_attribute_metrics(
        extraction_run_id="run_1",
        calculated_metrics=[
            AttributeCountMetric(
                attribute=attribute,
                metric_name="n_gold_instances",
                value=10,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy",
                value=0.6,
                extraction_run_id="run_1",
            ),
        ],
    )
    json_path = tmp_path / "metrics.json"
    report.to_json(json_path)
    loaded = RunMetricsReport.from_json(json_path)
    assert loaded.model_dump(mode="json") == report.model_dump(mode="json")


def test_run_metrics_report_to_json_omits_null_scores(tmp_path) -> None:
    """JSON serialisation drops None score keys; counts stay integers."""
    attribute = Attribute(
        attribute_id=1,
        attribute_label="Outcome label",
        output_data_type=AttributeType.STRING,
    )
    report = RunMetricsReport.from_attribute_metrics(
        extraction_run_id="run_1",
        calculated_metrics=[
            AttributeCountMetric(
                attribute=attribute,
                metric_name="n_gold_instances",
                value=10,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy",
                value=0.6,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy_given_good_source",
                value=None,
                extraction_run_id="run_1",
            ),
        ],
    )
    assert report.format_version == 1
    assert report.attributes[0].attribute_type == AttributeType.STRING
    assert report.attributes[0].counts == {"n_gold_instances": 10}
    assert report.attributes[0].metrics["accuracy"] == 0.6
    assert report.attributes[0].metrics["accuracy_given_good_source"] is None

    json_path = tmp_path / "metrics.json"
    report.to_json(json_path)
    loaded = RunMetricsReport.from_json(json_path)
    assert loaded.attributes[0].counts == {"n_gold_instances": 10}
    assert loaded.attributes[0].metrics == {"accuracy": 0.6}
    assert "accuracy_given_good_source" not in loaded.attributes[0].metrics


def test_run_metrics_report_to_csv_writes_blank_for_none(tmp_path) -> None:
    """Wide CSV leaves blank cells for None scores and integers for counts."""
    attribute = Attribute(
        attribute_id=1,
        attribute_label="Outcome label",
        output_data_type=AttributeType.STRING,
    )
    report = RunMetricsReport.from_attribute_metrics(
        extraction_run_id="run_1",
        calculated_metrics=[
            AttributeCountMetric(
                attribute=attribute,
                metric_name="n_gold_instances",
                value=10,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy",
                value=0.6,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy_given_good_source",
                value=None,
                extraction_run_id="run_1",
            ),
        ],
    )
    csv_path = tmp_path / "metrics.csv"
    report.to_csv(csv_path)
    row = next(csv.DictReader(csv_path.open()))
    assert row["n_gold_instances"] == "10"
    assert row["accuracy"] == "0.6"
    assert row["accuracy_given_good_source"] == ""
