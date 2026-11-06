"""Functions for evaluating hierarchical extraction predictions against the EPPI gold standard."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from dotenv import load_dotenv

from deet.hierarchical_mvp.evaluation_helpers_hierarchical import (
    GOLD_ARMS_SHEET,
    PREDICTION_INTERVENTIONS_SHEET,
    EvaluationCandidate,
    EvaluationSupportValue,
    classify_field,
    load_reference_mapping,
    match_evaluation_rows,
    read_xlsx_sheet_as_dicts,
)
from deet.hierarchical_mvp.utils import (
    EXTRACTION_ID_COLUMN,
    _open_csv_for_write,
    configure_lm,
)
from deet.logger import logger

_DEFAULT_LLM_MODEL = "azure/gpt-5.6-terra"
_DEFAULT_MAX_TOKENS = 4000


class EvaluationSource(TypedDict):
    """Sheet and column containing one side of an evaluation."""

    sheet: str
    column: str
    reference_column: NotRequired[str]


class SupportColumnPair(TypedDict):
    """Equivalent gold and prediction columns used to help match rows."""

    gold_column: str
    prediction_column: str


class EvaluationMap(TypedDict):
    """Field locations, matching support, and independently selected CSV context."""

    gold: EvaluationSource
    prediction: EvaluationSource
    support_columns: NotRequired[list[SupportColumnPair]]
    gold_context: NotRequired[list[str]]
    prediction_context: NotRequired[list[str]]


INTERVENTION_EVALUATION_MAP: EvaluationMap = {
    "gold": {"sheet": GOLD_ARMS_SHEET, "column": "title"},
    "prediction": {
        "sheet": PREDICTION_INTERVENTIONS_SHEET,
        "column": "group_name",
    },
}


def _read_evaluation_rows(
    xlsx_path: Path,
    sheet_name: str,
    columns: set[str],
) -> list[dict[str, Any]]:
    rows = read_xlsx_sheet_as_dicts(
        xlsx_path,
        sheet_name,
        required_columns=columns - {EXTRACTION_ID_COLUMN},
    )
    return [
        {
            **row,
            EXTRACTION_ID_COLUMN: (
                index
                if row.get(EXTRACTION_ID_COLUMN) in (None, "")
                else row[EXTRACTION_ID_COLUMN]
            ),
        }
        for index, row in enumerate(rows, start=1)
    ]


def _load_prediction_rows(
    mapping_rows: list[dict[str, str]],
    prediction_source: EvaluationSource,
    columns: set[str],
) -> dict[Path, list[dict[str, Any]]]:
    rows_by_path: dict[Path, list[dict[str, Any]]] = {}
    for mapping_row in mapping_rows:
        extraction_path = Path(mapping_row["extraction_xlsx_path"])
        if not extraction_path.exists():
            logger.warning(
                f"Skipping reference {mapping_row['reference_item_id']}: "
                f"extraction file not found at {extraction_path}"
            )
            continue
        if extraction_path not in rows_by_path:
            rows_by_path[extraction_path] = _read_evaluation_rows(
                extraction_path, prediction_source["sheet"], columns
            )
    return rows_by_path


def _build_candidates(
    rows: list[dict[str, Any]],
    value_column: str,
    support_column_pairs: list[SupportColumnPair],
    *,
    gold: bool,
) -> list[EvaluationCandidate]:
    candidates: list[EvaluationCandidate] = []
    for index, row in enumerate(rows):
        support = [
            EvaluationSupportValue(
                gold_column=pair["gold_column"],
                prediction_column=pair["prediction_column"],
                value=str(
                    row.get(pair["gold_column"] if gold else pair["prediction_column"])
                    or ""
                ).strip(),
            )
            for pair in support_column_pairs
        ]
        candidates.append(
            EvaluationCandidate(
                index=index,
                value=str(row.get(value_column) or "").strip(),
                support=support,
            )
        )
    return candidates


def evaluate_fields(  # noqa: PLR0913
    mapping_csv_path: str | Path,
    gold_xlsx_path: str | Path,
    output_csv_path: str | Path,
    eval_map: EvaluationMap,
    llm_model: str = _DEFAULT_LLM_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Path:
    """
    Evaluate configured prediction values against configured gold values.

    For every reference listed in the mapping CSV (see
    `evaluation_helpers_hierarchical.generate_reference_mapping_template`), rows from
    the configured gold and prediction sheets are paired via an LLM-as-judge step.
    Optional support-column pairs are supplied to the judge as additional context.
    gold_context/prediction_context are exported only, without affecting matching.
    Source row identifiers use Extraction_ID or a 1-based non-empty entry number
    assigned across the whole sheet before grouping by reference.
    """
    mapping_csv_path = Path(mapping_csv_path)
    gold_xlsx_path = Path(gold_xlsx_path)
    output_csv_path = Path(output_csv_path)

    gold_source = eval_map["gold"]
    prediction_source = eval_map["prediction"]
    support_columns = eval_map.get("support_columns", [])
    gold_context = list(dict.fromkeys(eval_map.get("gold_context", [])))
    prediction_context = list(dict.fromkeys(eval_map.get("prediction_context", [])))
    gold_support_columns = {pair["gold_column"] for pair in support_columns}
    prediction_support_columns = {pair["prediction_column"] for pair in support_columns}
    gold_reference_column = gold_source.get("reference_column", "reference_item_id")

    mapping_rows = load_reference_mapping(mapping_csv_path)

    gold_rows = _read_evaluation_rows(
        gold_xlsx_path,
        gold_source["sheet"],
        {
            gold_reference_column,
            gold_source["column"],
            *gold_support_columns,
            *gold_context,
        },
    )
    gold_rows_by_reference: dict[str, list[dict[str, Any]]] = {}
    for row in gold_rows:
        reference_id = str(row.get(gold_reference_column))
        gold_rows_by_reference.setdefault(reference_id, []).append(row)

    prediction_rows_by_path = _load_prediction_rows(
        mapping_rows,
        prediction_source,
        {prediction_source["column"], *prediction_support_columns, *prediction_context},
    )

    load_dotenv()
    configure_lm(llm_model, max_tokens)

    results: list[dict[str, Any]] = []
    for mapping_row in mapping_rows:
        reference_item_id = mapping_row["reference_item_id"]
        extraction_xlsx_path = Path(mapping_row["extraction_xlsx_path"])
        if extraction_xlsx_path not in prediction_rows_by_path:
            continue

        reference_gold_rows = gold_rows_by_reference.get(str(reference_item_id), [])
        predicted_rows = prediction_rows_by_path[extraction_xlsx_path]

        gold_candidates = _build_candidates(
            reference_gold_rows,
            gold_source["column"],
            support_columns,
            gold=True,
        )
        predicted_candidates = _build_candidates(
            predicted_rows,
            prediction_source["column"],
            support_columns,
            gold=False,
        )

        matches = match_evaluation_rows(predicted_candidates, gold_candidates)
        pairs = ", ".join(
            f"{predicted_candidates[match.predicted_index].value!r} -> "
            f"{gold_candidates[match.matched_gold_index].value!r}"
            if match.matched_gold_index is not None
            else (
                f"{predicted_candidates[match.predicted_index].value!r} -> '(no match)'"
            )
            for match in matches
        )
        match_log_line = (
            f"Reference {reference_item_id}: matched rows: {pairs or '(none)'}"
        )
        logger.info(match_log_line)
        print(match_log_line)  # logger only writes to deet.log, not the terminal
        matched_gold_indexes = {
            match.matched_gold_index
            for match in matches
            if match.matched_gold_index is not None
        }
        row_pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = [
            (
                reference_gold_rows[match.matched_gold_index]
                if match.matched_gold_index is not None
                else None,
                predicted_rows[match.predicted_index],
            )
            for match in matches
        ]
        row_pairs.extend(
            (gold_row, None)
            for gold_index, gold_row in enumerate(reference_gold_rows)
            if gold_index not in matched_gold_indexes
        )

        for gold_row, predicted_row in row_pairs:
            gold_value = gold_row.get(gold_source["column"]) if gold_row else None
            predicted_value = (
                predicted_row.get(prediction_source["column"])
                if predicted_row
                else None
            )
            field_result = classify_field(
                gold_value=gold_value,
                predicted_value=predicted_value,
            )
            results.append(
                {
                    "reference_item_id": reference_item_id,
                    "predicted_value": predicted_value,
                    "gold_value": gold_value,
                    "classification": field_result.classification,
                    "exact_match": field_result.exact_match,
                    "fuzzy_score": field_result.fuzzy_score,
                    "gold_file": str(gold_xlsx_path) if gold_row else "",
                    "prediction_file": (
                        str(extraction_xlsx_path) if predicted_row else ""
                    ),
                    "gold_sheet": gold_source["sheet"] if gold_row else "",
                    "prediction_sheet": (
                        prediction_source["sheet"] if predicted_row else ""
                    ),
                    "gold_excel_row": gold_row[EXTRACTION_ID_COLUMN]
                    if gold_row
                    else "",
                    "prediction_excel_row": (
                        predicted_row[EXTRACTION_ID_COLUMN] if predicted_row else ""
                    ),
                    "LLM_judge_status": (
                        "prediction_only"
                        if gold_row is None
                        else "gold_only"
                        if predicted_row is None
                        else "matched"
                    ),
                    **{
                        f"gold_context__{column}": (gold_row or {}).get(column, "")
                        for column in gold_context
                    },
                    **{
                        f"prediction_context__{column}": (predicted_row or {}).get(
                            column, ""
                        )
                        for column in prediction_context
                    },
                }
            )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reference_item_id",
        "predicted_value",
        "gold_value",
        "classification",
        "exact_match",
        "fuzzy_score",
        "gold_file",
        "prediction_file",
        "gold_sheet",
        "prediction_sheet",
        "gold_excel_row",
        "prediction_excel_row",
        "LLM_judge_status",
        *(f"gold_context__{column}" for column in gold_context),
        *(f"prediction_context__{column}" for column in prediction_context),
    ]
    with _open_csv_for_write(output_csv_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Evaluation written to {output_csv_path} ({len(results)} rows)")
    return output_csv_path


def evaluate_interventions(
    mapping_csv_path: str | Path,
    gold_xlsx_path: str | Path,
    output_csv_path: str | Path,
    llm_model: str = _DEFAULT_LLM_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Path:
    """Evaluate intervention group names using the legacy default column mapping."""
    return evaluate_fields(
        mapping_csv_path,
        gold_xlsx_path,
        output_csv_path,
        INTERVENTION_EVALUATION_MAP,
        llm_model,
        max_tokens,
    )


def summarize_evaluation(evaluation_csv_path: str | Path) -> dict[str, float]:
    """Summarize a TP/FP/FN/TN evaluation CSV (see `evaluate_interventions`) into scores.

    Returns a dict with the raw `TP`/`FP`/`FN`/`TN` counts plus `precision`, `recall`
    and `f1` (each `0.0` when their denominator is zero).
    """
    evaluation_csv_path = Path(evaluation_csv_path)
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    with evaluation_csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            classification = row.get("classification", "")
            if classification in counts:
                counts[classification] += 1

    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {**counts, "precision": precision, "recall": recall, "f1": f1}
