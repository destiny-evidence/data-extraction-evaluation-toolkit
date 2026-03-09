from loguru import logger

from deet.evaluators.gold_standard_llm_evaluator import GoldStandardLLMEvaluator

pytest_plugins = ["tests.unit.test_eppi"]


def test_evaluator_evaluates(processed_data):
    evaluator = GoldStandardLLMEvaluator(
        gold_standard_annotated_documents=processed_data.annotated_documents,
        llm_annotated_documents=processed_data.annotated_documents,
        attributes=[processed_data.attributes[0]],
        extraction_run_id="test_run",
    )
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
