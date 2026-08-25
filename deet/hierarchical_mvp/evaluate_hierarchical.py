"""Functions for evaluating hierarchical extraction predictions against the EPPI gold standard."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deet.hierarchical_mvp.evaluation_helpers_hierarchical import (
    GOLD_ARMS_SHEET,
    PREDICTION_INTERVENTIONS_SHEET,
    classify_field,
    load_reference_mapping,
    match_predicted_to_gold_arms,
    read_xlsx_sheet_as_dicts,
)
from deet.hierarchical_mvp.utils import _open_csv_for_write, configure_lm
from deet.logger import logger

_DEFAULT_LLM_MODEL = "azure/gpt-5.6-terra"
_DEFAULT_MAX_TOKENS = 4000


def evaluate_interventions(
    mapping_csv_path: str | Path,
    gold_xlsx_path: str | Path,
    output_csv_path: str | Path,
    llm_model: str = _DEFAULT_LLM_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> Path:
    """Evaluate predicted 'interventions.group_name' against gold 'Arms.title'.

    For every reference listed in the mapping CSV (see
    `evaluation_helpers_hierarchical.generate_reference_mapping_template`), gold arm
    titles are paired with predicted intervention group names via an LLM-as-judge step
    (each gold arm matched to at most one prediction), then every matched pair - plus
    any unmatched predicted or gold row - is classified as TP/FP/FN/TN.

    This starts with just the `group_name`/`title` field; the row-level matches it
    produces are meant to be reused to evaluate the other Intervention/Outcome columns
    against their gold counterparts.
    """
    mapping_csv_path = Path(mapping_csv_path)
    gold_xlsx_path = Path(gold_xlsx_path)
    output_csv_path = Path(output_csv_path)

    load_dotenv()
    configure_lm(llm_model, max_tokens)

    mapping_rows = load_reference_mapping(mapping_csv_path)

    gold_arms_by_reference: dict[str, list[dict[str, Any]]] = {}
    for row in read_xlsx_sheet_as_dicts(gold_xlsx_path, GOLD_ARMS_SHEET):
        reference_id = str(row.get("reference_item_id"))
        gold_arms_by_reference.setdefault(reference_id, []).append(row)

    results: list[dict[str, Any]] = []
    for mapping_row in mapping_rows:
        reference_item_id = mapping_row["reference_item_id"]
        extraction_xlsx_path = Path(mapping_row["extraction_xlsx_path"])
        if not extraction_xlsx_path.exists():
            logger.warning(
                f"Skipping reference {reference_item_id}: extraction file not found "
                f"at {extraction_xlsx_path}"
            )
            continue

        reference_gold_arms = gold_arms_by_reference.get(str(reference_item_id), [])
        predicted_interventions = read_xlsx_sheet_as_dicts(
            extraction_xlsx_path, PREDICTION_INTERVENTIONS_SHEET
        )

        gold_titles = [str(row.get("title") or "").strip() for row in reference_gold_arms]
        predicted_names = [
            str(row.get("group_name") or "").strip() for row in predicted_interventions
        ]

        matches = match_predicted_to_gold_arms(predicted_names, gold_titles)
        pairs = ", ".join(
            f"{match.predicted_group_name!r} -> {match.matched_gold_title or '(no match)'!r}"
            for match in matches
        )
        match_log_line = f"Reference {reference_item_id}: matched arms: {pairs or '(none)'}"
        logger.info(match_log_line)
        print(match_log_line)  # logger only writes to deet.log, not the terminal
        matched_gold_titles: set[str] = set()

        for match, predicted_row in zip(matches, predicted_interventions):
            gold_title = match.matched_gold_title.strip()
            gold_row = next(
                (
                    row
                    for row in reference_gold_arms
                    if str(row.get("title") or "").strip() == gold_title
                ),
                None,
            )
            if gold_row is not None:
                matched_gold_titles.add(gold_title)

            field_result = classify_field(
                gold_value=gold_row.get("title") if gold_row else None,
                predicted_value=predicted_row.get("group_name"),
            )
            results.append(
                {
                    "reference_item_id": reference_item_id,
                    "predicted_group_name": predicted_row.get("group_name"),
                    "gold_title": gold_row.get("title") if gold_row else "",
                    "classification": field_result.classification,
                    "exact_match": field_result.exact_match,
                    "fuzzy_score": field_result.fuzzy_score,
                }
            )

        # Gold arms never claimed by a predicted match are missed extractions (FN).
        for gold_row in reference_gold_arms:
            title = str(gold_row.get("title") or "").strip()
            if title and title not in matched_gold_titles:
                field_result = classify_field(gold_value=gold_row.get("title"), predicted_value=None)
                results.append(
                    {
                        "reference_item_id": reference_item_id,
                        "predicted_group_name": "",
                        "gold_title": gold_row.get("title"),
                        "classification": field_result.classification,
                        "exact_match": field_result.exact_match,
                        "fuzzy_score": field_result.fuzzy_score,
                    }
                )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reference_item_id",
        "predicted_group_name",
        "gold_title",
        "classification",
        "exact_match",
        "fuzzy_score",
    ]
    with _open_csv_for_write(output_csv_path) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Intervention evaluation written to {output_csv_path} ({len(results)} rows)")
    return output_csv_path


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
