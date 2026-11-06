import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openpyxl import load_workbook

from deet.hierarchical_mvp.evaluation_helpers_hierarchical import (
    EvaluationCandidate,
    EvaluationMatch,
    EvaluationSupportValue,
    export_from_eppi,
    match_evaluation_rows,
    read_xlsx_sheet_as_dicts,
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


def test_eppi_export_numbers_entries_per_sheet(tmp_path):
    path = tmp_path / "eppi.json"
    path.write_text(
        json.dumps(
            {
                "References": [
                    {
                        "ItemId": 42,
                        "Title": "Trial A",
                        "Arms": [{"title": "A"}, {"title": "B"}],
                        "Outcomes": [{"Title": "Weight"}],
                    },
                    {
                        "ItemId": 43,
                        "Title": "Trial B",
                        "Arms": [{"title": "C"}],
                        "Outcomes": [{"Title": "Weight"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    output = export_from_eppi(path)
    workbook = load_workbook(output)
    try:
        for sheet in workbook:
            values = list(sheet.values)
            assert values[0][0] == "Extraction_ID"
            assert [row[0] for row in values[1:]] == list(range(1, len(values)))
        assert workbook["Arms"].max_row == 4
        assert workbook["Outcomes"].max_row == 3
        assert list(workbook["Timepoints"].values) == [("Extraction_ID",)]
    finally:
        workbook.close()
    rows = read_xlsx_sheet_as_dicts(output, "Arms", required_columns={"title"})
    assert [row["Extraction_ID"] for row in rows] == [1, 2, 3]
    with pytest.raises(KeyError, match="Column\\(s\\) missing not found"):
        read_xlsx_sheet_as_dicts(output, "Timepoints", required_columns={"missing"})
