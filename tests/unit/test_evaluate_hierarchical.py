import csv
from unittest.mock import patch

import pytest
from openpyxl import Workbook

from deet.hierarchical_mvp.evaluate_hierarchical import (
    _read_evaluation_rows,
    evaluate_fields,
    evaluate_interventions,
    summarize_evaluation,
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

    def read_sheet(path, sheet, *, required_columns):
        assert required_columns
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


def _write_sheet(path, name, headers, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = name
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()


def _write_mapping(path, predictions):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["reference_item_id", "extraction_xlsx_path"])
        writer.writerows(predictions)


@pytest.mark.parametrize("with_ids", [False, True])
def test_evaluation_exports_context_ids_and_judge_status(tmp_path, with_ids):
    gold_path = tmp_path / "gold.xlsx"
    prediction_path = tmp_path / "prediction.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "evaluation.csv"
    gold_headers = ["reference_item_id", "Title", "Arm", "count", "enabled"]
    prediction_headers = ["name", "group"]
    gold_rows = [
        ["99", "Weight", "Another study", 5, True],
        ["42", "Weight", "Arm A", 0, False],
        [],
        ["42", "Weight", "Arm B", 10, True],
        ["42", "Unextracted", "Arm C", None, None],
    ]
    prediction_rows = [["Weight", "Arm B"], [], ["Weight", "Arm A"], ["Extra", "Arm D"]]
    if with_ids:
        gold_headers.append("Extraction_ID")
        prediction_headers.append("Extraction_ID")
        for row, extraction_id in zip(gold_rows, [60, 70, None, 80, 90], strict=True):
            if row:
                row.append(extraction_id)
        for row, extraction_id in zip(
            prediction_rows, [900, None, 800, 700], strict=True
        ):
            if row:
                row.append(extraction_id)
    _write_sheet(gold_path, "Outcomes", gold_headers, gold_rows)
    _write_sheet(prediction_path, "outcomes", prediction_headers, prediction_rows)
    original_workbooks = (gold_path.read_bytes(), prediction_path.read_bytes())
    _write_mapping(mapping_path, [("42", prediction_path)])
    eval_map = {
        "gold": {"sheet": "Outcomes", "column": "Title"},
        "prediction": {"sheet": "outcomes", "column": "name"},
        "support_columns": [{"gold_column": "Arm", "prediction_column": "group"}],
        "gold_context": ["Arm", "count", "enabled", "Extraction_ID", "Arm"],
        "prediction_context": ["group", "Extraction_ID"],
    }

    with (
        patch("deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.match_evaluation_rows",
            return_value=[
                EvaluationMatch(predicted_index=0, matched_gold_index=1),
                EvaluationMatch(predicted_index=1, matched_gold_index=0),
                EvaluationMatch(predicted_index=2),
            ],
        ) as match_rows,
    ):
        evaluate_fields(mapping_path, gold_path, output_path, eval_map)

    with output_path.open(encoding="utf-8-sig", newline="") as output:
        reader = csv.DictReader(output)
        rows = list(reader)
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
    assert [row["LLM_judge_status"] for row in rows] == [
        "matched",
        "matched",
        "prediction_only",
        "gold_only",
    ]
    assert [row["gold_excel_row"] for row in rows] == (
        ["80", "70", "", "90"] if with_ids else ["3", "2", "", "4"]
    )
    assert [row["prediction_excel_row"] for row in rows] == (
        ["900", "800", "700", ""] if with_ids else ["1", "2", "3", ""]
    )
    assert [row["gold_context__Arm"] for row in rows] == ["Arm B", "Arm A", "", "Arm C"]
    assert [row["prediction_context__group"] for row in rows] == [
        "Arm B",
        "Arm A",
        "Arm D",
        "",
    ]
    assert rows[1]["gold_context__count"] == "0"
    assert rows[1]["gold_context__enabled"] == "False"
    assert rows[0]["gold_context__Extraction_ID"] == rows[0]["gold_excel_row"]
    assert (
        rows[0]["prediction_context__Extraction_ID"] == rows[0]["prediction_excel_row"]
    )
    assert rows[0]["gold_file"] == str(gold_path)
    assert rows[0]["prediction_file"] == str(prediction_path)
    assert rows[0]["gold_sheet"] == "Outcomes"
    assert rows[0]["prediction_sheet"] == "outcomes"
    assert rows[2]["gold_file"] == rows[2]["gold_sheet"] == ""
    assert rows[3]["prediction_file"] == rows[3]["prediction_sheet"] == ""
    predicted_candidates, gold_candidates = match_rows.call_args.args
    assert [candidate.index for candidate in gold_candidates] == [0, 1, 2]
    assert all(len(candidate.support) == 1 for candidate in gold_candidates)
    assert predicted_candidates[0].support[0].value == "Arm B"
    assert gold_candidates[0].support[0].value == "Arm A"
    scores = summarize_evaluation(output_path)
    assert {key: scores[key] for key in ("TP", "FP", "FN", "TN")} == {
        "TP": 2,
        "FP": 1,
        "FN": 1,
        "TN": 0,
    }
    assert (gold_path.read_bytes(), prediction_path.read_bytes()) == original_workbooks


@pytest.mark.parametrize("side", ["gold", "prediction"])
@pytest.mark.parametrize("header_only", [False, True])
def test_missing_context_fails_before_any_llm_calls(tmp_path, side, header_only):
    gold_path = tmp_path / "gold.xlsx"
    first_prediction = tmp_path / "first.xlsx"
    second_prediction = tmp_path / "second.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "evaluation.csv"
    _write_sheet(
        gold_path,
        "Outcomes",
        ["reference_item_id", "Title"],
        [] if header_only else [["42", "Weight"]],
    )
    _write_sheet(first_prediction, "outcomes", ["name", "group"], [["Weight", "A"]])
    _write_sheet(
        second_prediction,
        "outcomes",
        ["name"],
        [] if header_only else [["Weight"]],
    )
    _write_mapping(mapping_path, [("42", first_prediction), ("43", second_prediction)])
    eval_map = {
        "gold": {"sheet": "Outcomes", "column": "Title"},
        "prediction": {"sheet": "outcomes", "column": "name"},
        f"{side}_context": ["group"],
    }
    with (
        patch("deet.hierarchical_mvp.evaluate_hierarchical.configure_lm") as configure,
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.match_evaluation_rows"
        ) as judge,
        pytest.raises(KeyError, match="Column\\(s\\) group not found") as error,
    ):
        evaluate_fields(mapping_path, gold_path, output_path, eval_map)
    assert (
        str(gold_path if side == "gold" else second_prediction) in error.value.args[0]
    )
    configure.assert_not_called()
    judge.assert_not_called()
    assert not output_path.exists()


def test_empty_evaluation_has_context_headers(tmp_path):
    gold_path = tmp_path / "gold.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "evaluation.csv"
    _write_sheet(gold_path, "Outcomes", ["reference_item_id", "Title", "Arm"], [])
    _write_mapping(mapping_path, [])
    with patch("deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"):
        evaluate_fields(
            mapping_path,
            gold_path,
            output_path,
            {
                "gold": {"sheet": "Outcomes", "column": "Title"},
                "prediction": {"sheet": "outcomes", "column": "name"},
                "gold_context": ["Arm", "Extraction_ID"],
                "prediction_context": ["group", "Extraction_ID"],
            },
        )
    with output_path.open(encoding="utf-8-sig", newline="") as output:
        reader = csv.DictReader(output)
        assert list(reader) == []
        assert reader.fieldnames[-4:] == [
            "gold_context__Arm",
            "gold_context__Extraction_ID",
            "prediction_context__group",
            "prediction_context__Extraction_ID",
        ]


def test_blank_extraction_ids_fall_back_without_overwriting_existing_ids(tmp_path):
    path = tmp_path / "legacy.xlsx"
    _write_sheet(
        path,
        "outcomes",
        ["Extraction_ID", "name"],
        [[100, "Weight"], [None, "Weight"], [], ["OUT-3", "Weight"]],
    )
    rows = _read_evaluation_rows(path, "outcomes", {"name", "Extraction_ID"})
    assert [row["Extraction_ID"] for row in rows] == [100, 2, "OUT-3"]


@pytest.mark.parametrize("empty_side", ["gold", "prediction"])
def test_evaluation_with_one_empty_sheet(tmp_path, empty_side):
    gold_path = tmp_path / "gold.xlsx"
    prediction_path = tmp_path / "prediction.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    output_path = tmp_path / "evaluation.csv"
    _write_sheet(
        gold_path,
        "Outcomes",
        ["reference_item_id", "Title", "context"],
        [] if empty_side == "gold" else [["42", "Weight", "Gold arm"]],
    )
    _write_sheet(
        prediction_path,
        "outcomes",
        ["name", "context"],
        [] if empty_side == "prediction" else [["Weight", "Predicted arm"]],
    )
    _write_mapping(mapping_path, [("42", prediction_path)])
    with patch("deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"):
        evaluate_fields(
            mapping_path,
            gold_path,
            output_path,
            {
                "gold": {"sheet": "Outcomes", "column": "Title"},
                "prediction": {"sheet": "outcomes", "column": "name"},
                "gold_context": ["context"],
                "prediction_context": ["context"],
            },
        )
    with output_path.open(encoding="utf-8-sig", newline="") as output:
        rows = list(csv.DictReader(output))
    assert len(rows) == 1
    expected_status = "prediction_only" if empty_side == "gold" else "gold_only"
    assert rows[0]["LLM_judge_status"] == expected_status
    assert rows[0]["classification"] == ("FP" if empty_side == "gold" else "FN")
    assert rows[0][f"{empty_side}_excel_row"] == ""
    assert rows[0][f"{empty_side}_context__context"] == ""
