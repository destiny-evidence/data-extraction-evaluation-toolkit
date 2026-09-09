import csv
from unittest.mock import patch

from deet.hierarchical_mvp.evaluate_hierarchical import (
    evaluate_fields,
    evaluate_interventions,
)
from deet.hierarchical_mvp.evaluation_helpers_hierarchical import EvaluationMatch


def test_evaluate_interventions_configures_anthropic_judge_model(tmp_path):
    output_path = tmp_path / "evaluation.csv"

    with (
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"
        ) as configure_lm,
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.load_reference_mapping",
            return_value=[],
        ),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.read_xlsx_sheet_as_dicts",
            return_value=[],
        ),
    ):
        evaluate_interventions(
            "mapping.csv",
            "gold.xlsx",
            output_path,
            llm_model="anthropic/claude-sonnet",
            max_tokens=4096,
        )

    configure_lm.assert_called_once_with("anthropic/claude-sonnet", 4096)


def test_evaluate_fields_uses_configured_columns_and_support(tmp_path):
    prediction_path = tmp_path / "prediction.xlsx"
    prediction_path.touch()
    output_path = tmp_path / "evaluation.csv"
    eval_map = {
        "gold": {"sheet": "Gold outcomes", "column": "outcome_name"},
        "prediction": {"sheet": "outcomes", "column": "name"},
        "support_columns": [
            {
                "gold_column": "gold_definition",
                "prediction_column": "predicted_definition",
            }
        ],
    }
    gold_rows = [
        {
            "reference_item_id": "42",
            "outcome_name": "Body mass index",
            "gold_definition": "BMI after 12 months",
        }
    ]
    predicted_rows = [
        {
            "name": "BMI",
            "predicted_definition": "Measured at twelve months",
        }
    ]

    def read_sheet(path, sheet):
        if sheet == "Gold outcomes":
            return gold_rows
        assert path == prediction_path
        assert sheet == "outcomes"
        return predicted_rows

    with (
        patch("deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.load_reference_mapping",
            return_value=[
                {
                    "reference_item_id": "42",
                    "extraction_xlsx_path": str(prediction_path),
                }
            ],
        ),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.read_xlsx_sheet_as_dicts",
            side_effect=read_sheet,
        ),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.match_evaluation_rows",
            return_value=[EvaluationMatch(predicted_index=0, matched_gold_index=0)],
        ) as match_rows,
    ):
        evaluate_fields("mapping.csv", "gold.xlsx", output_path, eval_map)

    predicted_candidates, gold_candidates = match_rows.call_args.args
    assert predicted_candidates[0].value == "BMI"
    assert predicted_candidates[0].support[0].model_dump() == {
        "gold_column": "gold_definition",
        "prediction_column": "predicted_definition",
        "value": "Measured at twelve months",
    }
    assert gold_candidates[0].value == "Body mass index"
    assert gold_candidates[0].support[0].model_dump() == {
        "gold_column": "gold_definition",
        "prediction_column": "predicted_definition",
        "value": "BMI after 12 months",
    }
    with output_path.open(encoding="utf-8-sig") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 1
    assert rows[0]["reference_item_id"] == "42"
    assert rows[0]["predicted_value"] == "BMI"
    assert rows[0]["gold_value"] == "Body mass index"
    assert rows[0]["classification"] == "FP"
    assert rows[0]["exact_match"] == "False"
    assert float(rows[0]["fuzzy_score"]) > 0
