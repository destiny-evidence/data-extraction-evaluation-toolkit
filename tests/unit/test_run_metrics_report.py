"""Unit tests for the machine-readable metrics.json report models."""

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.evaluation import AttributeMetric, RunMetricsReport


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
        ],
    )
    json_path = tmp_path / "metrics.json"
    report.to_json(json_path)
    loaded = RunMetricsReport.from_json(json_path)
    assert loaded.model_dump(mode="json") == report.model_dump(mode="json")


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
