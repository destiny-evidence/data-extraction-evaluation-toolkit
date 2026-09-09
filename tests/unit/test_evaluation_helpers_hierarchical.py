from types import SimpleNamespace
from unittest.mock import patch

from deet.hierarchical_mvp.evaluation_helpers_hierarchical import (
    EvaluationCandidate,
    EvaluationMatch,
    EvaluationSupportValue,
    match_evaluation_rows,
)


def test_match_evaluation_rows_passes_support_context_to_llm():
    predicted = [
        EvaluationCandidate(
            index=0,
            value="Treatment A",
            support=[
                EvaluationSupportValue(
                    gold_column="reported_mean",
                    prediction_column="group_mean",
                    value="10 mg",
                )
            ],
        )
    ]
    gold = [
        EvaluationCandidate(
            index=0,
            value="Arm A",
            support=[
                EvaluationSupportValue(
                    gold_column="reported_mean",
                    prediction_column="group_mean",
                    value="10 milligrams",
                )
            ],
        )
    ]

    with patch(
        "deet.hierarchical_mvp.evaluation_helpers_hierarchical.dspy.Predict"
    ) as predict_class:
        predict_class.return_value.return_value = SimpleNamespace(
            matches=[EvaluationMatch(predicted_index=0, matched_gold_index=0)]
        )
        matches = match_evaluation_rows(predicted, gold)

    predict_class.return_value.assert_called_once_with(
        predicted_candidates=predicted,
        gold_candidates=gold,
    )
    assert matches == [EvaluationMatch(predicted_index=0, matched_gold_index=0)]
