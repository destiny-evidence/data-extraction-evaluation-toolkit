"""Unit tests for the machine-readable metrics.json report models."""

import csv

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.evaluation import (
    AttributeCountMetric,
    AttributeScoreMetric,
    RunMetricsReport,
    preferred_metric_column_names,
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


def test_preferred_metric_column_names_derived_from_registries() -> None:
    """Column order follows metric registries; source-fidelity types get suffixes."""
    columns = preferred_metric_column_names()
    assert columns[0] == "n_gold_instances"
    assert "accuracy" in columns
    assert "accuracy_given_good_source" in columns
    assert "accuracy_given_bad_source" in columns
    assert "edit_distance_match_rate_given_good_source" in columns
    assert "precision" in columns
    assert "precision_given_good_source" not in columns
    assert columns.index("accuracy") < columns.index("precision")


def test_run_metrics_report_to_csv_appends_custom_metrics_sorted(tmp_path) -> None:
    """Custom metric columns not in the preferred list are appended alphabetically."""
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
                value=1,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="accuracy",
                value=1.0,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="jaccard_score",
                value=0.5,
                extraction_run_id="run_1",
            ),
            AttributeScoreMetric(
                attribute=attribute,
                metric_name="jaccard_score_given_good_source",
                value=0.5,
                extraction_run_id="run_1",
            ),
        ],
    )
    csv_path = tmp_path / "metrics.csv"
    report.to_csv(csv_path)
    fieldnames = next(csv.DictReader(csv_path.open())).keys()
    names = list(fieldnames)
    assert "jaccard_score" in names
    assert "jaccard_score_given_good_source" in names
    assert names.index("accuracy") < names.index("jaccard_score")
