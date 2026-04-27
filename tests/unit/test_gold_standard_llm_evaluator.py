import csv
from copy import deepcopy
from unittest.mock import patch

import pytest
from loguru import logger
from rich.table import Table

from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator

pytest_plugins = ["tests.unit.test_eppi"]


def test_evaluator_evaluates(processed_data):
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        extraction_run_id="test_run",
    )
    evaluator.evaluate_llm_annotations()
    for metric in evaluator.calculated_metrics:
        assert metric.value == 1


def test_evaluator_evaluates_with_custom_metric(processed_data):
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        custom_metrics=["jaccard_score"],
        extraction_run_id="test_run",
    )
    assert "jaccard_score" in evaluator.metrics_config
    evaluator.evaluate_llm_annotations()
    for metric in evaluator.calculated_metrics:
        assert metric.value == 1


def test_evaluator_evaluates_with_nonexistent_metric(processed_data):
    messages = []
    logger_id = logger.add(messages.append, level="WARNING")
    nonexistent_metric = "nonexistent_metric"
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        custom_metrics=[nonexistent_metric],
        extraction_run_id="test_run",
    )
    logger.remove(logger_id)
    assert any(f"Tried to add {nonexistent_metric}" in m for m in messages)
    assert "nonexistent_metric" not in evaluator.metrics_config
    evaluator.evaluate_llm_annotations()
    for metric in evaluator.calculated_metrics:
        assert metric.value == 1


def test_evaluator_evaluates_with_nonfloat_metric(processed_data):
    messages = []
    logger_id = logger.add(messages.append, level="WARNING")
    nonfloat_metric = "classification_report"
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        custom_metrics=[nonfloat_metric],
        extraction_run_id="",
    )
    logger.remove(logger_id)
    assert nonfloat_metric not in evaluator.metrics_config
    assert any(f"Tried to add {nonfloat_metric}" in m for m in messages)
    evaluator.evaluate_llm_annotations()
    for metric in evaluator.calculated_metrics:
        assert metric.value == 1


@pytest.fixture
def processed_data_missing_doc(processed_data):
    """Create ProcessedEppiAnnotationData with test attributes."""
    processed_data_missing_doc = deepcopy(processed_data)
    processed_data_missing_doc.annotated_documents = processed_data.annotated_documents[
        :-1
    ]
    return processed_data_missing_doc


# When a doc is missing from llm_preds, metrics should be None
# and we should warn rather than fail
def test_evaluator_fails_gracefully_missing_doc(
    processed_data, processed_data_missing_doc, tmp_path
):
    messages = []
    logger_id = logger.add(messages.append, level="WARNING")
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data_missing_doc.annotated_documents,
        attributes=[processed_data.attributes[0]],
        extraction_run_id="",
    )
    evaluator.evaluate_llm_annotations()
    for m in evaluator.calculated_metrics:
        if m.metric_name != "n_labels":
            assert m.value is None

    logger.remove(logger_id)
    assert any("LLM annotated doc not found" in m for m in messages)

    evaluator.export_llm_comparison(tmp_path / "llm_human_comparison.csv")


@pytest.fixture
def processed_data_duplicated_annotations(processed_data):
    """Create ProcessedEppiAnnotationData with test attributes."""
    processed_data_duplicated_annotations = deepcopy(processed_data)
    for doc in processed_data_duplicated_annotations.annotated_documents:
        doc.annotations = doc.annotations + doc.annotations
    return processed_data_duplicated_annotations


# When an llm returns multiple values for the same attribute, metrics should be None
# and we should warn rather than fail
def test_evaluator_fails_gracefully_duplicated_annotations(
    processed_data, processed_data_duplicated_annotations, tmp_path
):
    messages = []
    logger_id = logger.add(messages.append, level="WARNING")
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data_duplicated_annotations.annotated_documents,
        attributes=[processed_data.attributes[0]],
        extraction_run_id="",
    )
    evaluator.evaluate_llm_annotations()
    for m in evaluator.calculated_metrics:
        if m.metric_name != "n_labels":
            assert m.value is None

    logger.remove(logger_id)
    warn_string = "LLM produced multiple annotations for a single attribute"
    assert any(warn_string in m for m in messages)

    evaluator.export_llm_comparison(tmp_path / "llm_human_comparison.csv")


@pytest.fixture
def evaluator_evaluated(processed_data):
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        extraction_run_id="",
    )
    evaluator.evaluate_llm_annotations()
    return evaluator


def test_evaluator_writes_metrics(evaluator_evaluated, tmp_path):
    metric_csv_path = tmp_path / "metrics.csv"
    evaluator_evaluated.write_metrics_to_csv(metric_csv_path)
    reader = csv.DictReader(metric_csv_path.open())
    rows = list(reader)
    for r in rows:
        assert float(r["value"]) == 1.0


def test_evaluator_writes_comparison(evaluator_evaluated, tmp_path):
    comparison_csv_path = tmp_path / "llm_human_comparison.csv"
    evaluator_evaluated.export_llm_comparison(comparison_csv_path)
    reader = csv.DictReader(comparison_csv_path.open())
    rows = list(reader)
    for r in rows:
        assert r["human_extraction"] == r["llm_extraction"]


def test_evaluator_displays_metrics(evaluator_evaluated):
    with patch(
        "deet.evaluators.gold_standard_llm_evaluator.Console.print"
    ) as mock_print:
        evaluator_evaluated.display_metrics()

    # ensure something was printed
    assert mock_print.call_count == 1

    table = mock_print.call_args[0][0]
    assert isinstance(table, Table)

    column_headers = [c.header for c in table.columns]
    assert column_headers[0] == "Attribute"
    for metric_name in evaluator_evaluated.metrics_config:
        assert metric_name in column_headers

    first_column_cells = table.columns[0]._cells
    assert len(first_column_cells) == len(evaluator_evaluated.attributes)

    metric_columns = table.columns[1:]
    for col in metric_columns:
        for cell in col._cells:
            assert float(cell) == 1.0
