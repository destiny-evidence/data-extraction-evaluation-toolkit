import csv
from copy import deepcopy
from unittest.mock import patch

import pytest
from destiny_sdk.references import ReferenceFileInput
from loguru import logger
from rich.table import Table

from deet.data_models.base import AnnotationType, AttributeType
from deet.data_models.eppi import (
    EppiAttribute,
    EppiAttributeSelectionType,
    EppiDocument,
    EppiGoldStandardAnnotatedDocument,
    EppiGoldStandardAnnotation,
    EppiItemAttributeFullTextDetails,
)
from deet.data_models.evaluation import COUNT_METRIC_NAMES, RunMetricsReport
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
        if metric.metric_name in COUNT_METRIC_NAMES:
            continue
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
        if metric.metric_name in COUNT_METRIC_NAMES:
            continue
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
        if metric.metric_name in COUNT_METRIC_NAMES:
            continue
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
        if metric.metric_name in COUNT_METRIC_NAMES:
            continue
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
        if m.metric_name in COUNT_METRIC_NAMES or m.metric_name == "n_labels":
            continue
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
        if m.metric_name in COUNT_METRIC_NAMES or m.metric_name == "n_labels":
            continue
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
    assert len(rows) == 1
    row = rows[0]
    assert row["attribute_type"] == "bool"
    assert float(row["n_gold_instances"]) > 0
    assert float(row["accuracy"]) == 1.0
    assert float(row["precision"]) == 1.0
    assert float(row["recall"]) == 1.0
    assert float(row["f1_score"]) == 1.0
    assert float(row["n_labels"]) == 1.0


def test_evaluator_writes_metrics_json(evaluator_evaluated, tmp_path):
    metrics_json_path = tmp_path / "metrics.json"
    evaluator_evaluated.write_metrics_to_json(metrics_json_path)
    report = RunMetricsReport.from_json(metrics_json_path)
    assert report.format_version == 1
    assert report.extraction_run_id == ""
    assert len(report.attributes) == 1
    attr = report.attributes[0]
    assert attr.attribute_type == AttributeType.BOOL
    assert attr.counts["n_gold_instances"] > 0
    assert attr.metrics["accuracy"] == 1.0


def test_evaluator_writes_comparison(evaluator_evaluated, tmp_path):
    comparison_csv_path = tmp_path / "llm_human_comparison.csv"
    evaluator_evaluated.export_llm_comparison(comparison_csv_path)
    raw_text = comparison_csv_path.read_text(encoding="utf-8")
    assert "\r\r\n" not in raw_text
    reader = csv.DictReader(comparison_csv_path.open())
    fieldnames = reader.fieldnames or []
    expected_header = [
        "document_id",
        "external_id",
        "document_name",
        "attribute_id",
        "attribute_label",
        "attribute_presence",
        "human_additional_text",
        "item_attribute_full_text_details",
        "citation_page",
        "citation_highlight_text",
        "human_extraction",
        "llm_extraction",
        "llm_reasoning",
        "llm_verbatim_text",
        "human_verbatim_fuzzy_match_pct",
        "llm_verbatim_fuzzy_match_pct",
        "gold_value_in_citation",
        "gold_value_in_context",
        "match_status",
        "extraction_run_id",
    ]
    assert list(fieldnames) == expected_header
    rows = list(reader)
    assert len(rows) > 0
    for r in rows:
        assert r["attribute_presence"] in ("True", "False")
        assert r["human_extraction"] == r["llm_extraction"]
        assert "human_verbatim_fuzzy_match_pct" in r
        assert "llm_verbatim_fuzzy_match_pct" in r
        assert "citation_page" in r
        assert "citation_highlight_text" in r
        assert "gold_value_in_citation" in r
        assert "gold_value_in_context" in r
        assert "match_status" in r
        # Hand-built fixture has no EPPI markup; citation fields stay empty.
        assert r["citation_page"] == ""
        assert r["citation_highlight_text"] == ""


def test_evaluator_comparison_exports_parsed_citations(tmp_path):
    """Comparison CSV gets citation_page / citation_highlight_text from EPPI markup."""
    attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=1,
        attribute_label="Attribute 1",
        output_data_type=AttributeType.BOOL,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    doc = EppiDocument(
        name="Doc 1", citation=ReferenceFileInput(), document_id=12345678
    )
    annotation = EppiGoldStandardAnnotation(
        attribute=attr,
        output_data=True,
        annotation_type=AnnotationType.HUMAN,
        additional_text="Dolor si amet...",
        item_attribute_full_text_details=[
            EppiItemAttributeFullTextDetails(
                item_document_id=423106,
                text='Page 1:\n[¬s]"Dolor si amet...[¬e]"',
            )
        ],
    )
    annotated_doc = EppiGoldStandardAnnotatedDocument(
        document=doc, annotations=[annotation]
    )
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=[annotated_doc],
        llm_annotated_documents=[annotated_doc],
        attributes=[attr],
        extraction_run_id="citation_test",
    )
    comparison_csv_path = tmp_path / "comparison.csv"
    evaluator.export_llm_comparison(comparison_csv_path)
    rows = list(csv.DictReader(comparison_csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["citation_page"] == "1"
    assert rows[0]["citation_highlight_text"] == "Dolor si amet..."
    assert "Page 1:" in rows[0]["item_attribute_full_text_details"]
    assert "[¬s]" in rows[0]["item_attribute_full_text_details"]


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
        if col.header in COUNT_METRIC_NAMES:
            continue
        for cell in col._cells:
            assert float(cell) == 1.0


def test_evaluator_mixed_types_include_extraction_metrics(tmp_path):
    """STRING / INTEGER / FLOAT attributes emit the new metric names in output."""
    string_attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=10,
        attribute_label="Outcome label",
        output_data_type=AttributeType.STRING,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    int_attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=11,
        attribute_label="Sample size",
        output_data_type=AttributeType.INTEGER,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    float_attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=12,
        attribute_label="Effect size",
        output_data_type=AttributeType.FLOAT,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    doc = EppiDocument(
        name="Mixed Doc", citation=ReferenceFileInput(), document_id=42424242
    )
    gold_doc = EppiGoldStandardAnnotatedDocument(
        document=doc,
        annotations=[
            EppiGoldStandardAnnotation(
                attribute=string_attr,
                output_data="hypertension",
                annotation_type=AnnotationType.HUMAN,
            ),
            EppiGoldStandardAnnotation(
                attribute=int_attr,
                output_data=100,
                annotation_type=AnnotationType.HUMAN,
            ),
            EppiGoldStandardAnnotation(
                attribute=float_attr,
                output_data=0.95,
                annotation_type=AnnotationType.HUMAN,
            ),
        ],
    )
    llm_doc = EppiGoldStandardAnnotatedDocument(
        document=doc,
        annotations=[
            EppiGoldStandardAnnotation(
                attribute=string_attr,
                output_data="hypertention",
                annotation_type=AnnotationType.LLM,
            ),
            EppiGoldStandardAnnotation(
                attribute=int_attr,
                output_data=101,
                annotation_type=AnnotationType.LLM,
            ),
            EppiGoldStandardAnnotation(
                attribute=float_attr,
                output_data=0.96,
                annotation_type=AnnotationType.LLM,
            ),
        ],
    )
    attributes = [string_attr, int_attr, float_attr]
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=[gold_doc],
        llm_annotated_documents=[llm_doc],
        attributes=attributes,
        extraction_run_id="mixed_types",
    )
    evaluator.evaluate_llm_annotations()

    metrics_by_attr: dict[str, dict[str, float | None]] = {}
    for metric in evaluator.calculated_metrics:
        label = metric.attribute.attribute_label
        metrics_by_attr.setdefault(label, {})[metric.metric_name] = metric.value

    assert "edit_distance_match_rate" in metrics_by_attr["Outcome label"]
    assert metrics_by_attr["Outcome label"]["edit_distance_match_rate"] == 1.0

    for numeric_label in ("Sample size", "Effect size"):
        assert "mean_absolute_error" in metrics_by_attr[numeric_label]
        assert "mean_absolute_percentage_error" in metrics_by_attr[numeric_label]
        assert metrics_by_attr[numeric_label]["mean_absolute_error"] is not None
        assert (
            metrics_by_attr[numeric_label]["mean_absolute_percentage_error"] is not None
        )

    metrics_csv = tmp_path / "metrics.csv"
    evaluator.write_metrics_to_csv(metrics_csv)
    rows = list(csv.DictReader(metrics_csv.open()))
    by_label = {row["attribute_label"]: row for row in rows}
    assert "Outcome label" in by_label
    assert "Sample size" in by_label
    assert "Effect size" in by_label
    assert by_label["Outcome label"]["edit_distance_match_rate"] != ""
    assert "accuracy_given_good_source" in by_label["Outcome label"]
    assert "accuracy_given_bad_source" in by_label["Outcome label"]
    # Fixture has no document context, so the good-source subset is empty.
    assert by_label["Outcome label"]["accuracy_given_good_source"] == ""
    assert by_label["Outcome label"]["edit_distance_match_rate_given_good_source"] == ""
    assert by_label["Outcome label"]["accuracy_given_bad_source"] != ""
    assert by_label["Sample size"]["mean_absolute_error"] != ""
    assert by_label["Sample size"]["mean_absolute_percentage_error"] != ""


def test_empty_bad_source_subset_leaves_metrics_blank(tmp_path) -> None:
    """When every instance is good-source, given_bad_source cells stay empty."""
    string_attr = EppiAttribute(  # type: ignore[call-arg]
        attribute_id=17896484,
        attribute_label="N random full",
        output_data_type=AttributeType.STRING,
        attribute_type=EppiAttributeSelectionType.INTERVENTION,
    )
    doc = EppiDocument(
        name="Good source doc",
        citation=ReferenceFileInput(),
        document_id=555,
        context="The sample size was 32 participants.",
    )
    gold_doc = EppiGoldStandardAnnotatedDocument(
        document=doc,
        annotations=[
            EppiGoldStandardAnnotation(  # type: ignore[call-arg]
                attribute=string_attr,
                output_data="32",
                annotation_type=AnnotationType.HUMAN,
            ),
        ],
    )
    llm_doc = EppiGoldStandardAnnotatedDocument(
        document=doc,
        annotations=[
            EppiGoldStandardAnnotation(  # type: ignore[call-arg]
                attribute=string_attr,
                output_data="32",
                annotation_type=AnnotationType.LLM,
            ),
        ],
    )
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=[gold_doc],
        llm_annotated_documents=[llm_doc],
        attributes=[string_attr],
        extraction_run_id="good_source_only",
    )
    evaluator.evaluate_llm_annotations()
    metrics_csv = tmp_path / "metrics.csv"
    evaluator.write_metrics_to_csv(metrics_csv)
    row = next(csv.DictReader(metrics_csv.open()))
    assert row["n_gold_instances"] == "1.0"
    assert row["n_good_source_instances"] == "1.0"
    assert row["edit_distance_match_rate"] == "1.0"
    assert row["edit_distance_match_rate_given_good_source"] == "1.0"
    assert row["accuracy_given_good_source"] == "1.0"
    assert row["edit_distance_match_rate_given_bad_source"] == ""
    assert row["accuracy_given_bad_source"] == ""

    comparison_csv = tmp_path / "comparison.csv"
    evaluator.export_llm_comparison(comparison_csv)
    comparison_row = next(csv.DictReader(comparison_csv.open()))
    assert comparison_row["match_status"] == "exact_match"
    assert comparison_row["gold_value_in_context"] == "True"
