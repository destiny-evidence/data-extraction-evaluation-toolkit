"""Post-processing utilities for Cochrane RCT extraction output.

After CochraneRCTExtractionPipeline runs and saves its JSON, use
``postprocess_cochrane_outcomes`` to convert that JSON into a split-row
CSV that matches the Study+results template format:

    Study | Outcome | Data type | Arm | Sample size | Cases | Mean | … | Footnotes

Each outcome in the JSON is stored with both arms' data.  This module
splits every outcome into *two rows* — one per arm — so that the output
is ready to be imported directly into RevMan / the Cochrane pipeline.

CLI usage (standalone):
    python -m deet.hierarchical_mvp.cochrane_postprocessing <json_path> [<output_csv>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deet.logger import logger

# ---------------------------------------------------------------------------
# Column order for the output CSV (matches Study+results template)
# ---------------------------------------------------------------------------

OUTPUT_FIELDNAMES: list[str] = [
    "Study",
    "Outcome",
    "Data type",
    "Arm",
    "Sample size",
    "Cases",
    "Mean",
    "SD",
    "SE",
    "Variance",
    "CI level",
    "CI start",
    "CI end",
    "t-test",
    "P value",
    "Footnotes",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _known_arm_names(study_dict: dict) -> set[str]:
    """Return the set of arm names declared in study.interventions."""
    return {
        (arm.get("arm") or "").strip()
        for arm in study_dict.get("interventions", [])
        if arm.get("arm")
    }


def _split_dichotomous(
    outcome: dict,
    study_id: str,
    known_arms: set[str],
) -> list[dict[str, str]]:
    """Split one dichotomous outcome dict into two arm-level row dicts."""
    name = (outcome.get("outcome_name") or "").strip()
    arm_a = (outcome.get("arm_a") or "").strip()
    arm_b = (outcome.get("arm_b") or "").strip()

    _warn_unknown_arms(name, arm_a, arm_b, known_arms)

    base = {"Study": study_id, "Outcome": name, "Data type": "Arm level"}
    return [
        {
            **base,
            "Arm": arm_a,
            "Sample size": str(outcome.get("sample_size_a") or ""),
            "Cases": str(outcome.get("cases_a") or ""),
            "P value": str(outcome.get("p_value") or ""),
            "Footnotes": str(outcome.get("footnotes") or ""),
        },
        {
            **base,
            "Arm": arm_b,
            "Sample size": str(outcome.get("sample_size_b") or ""),
            "Cases": str(outcome.get("cases_b") or ""),
            "P value": "",
            "Footnotes": "",
        },
    ]


def _split_continuous(
    outcome: dict,
    study_id: str,
    known_arms: set[str],
) -> list[dict[str, str]]:
    """Split one continuous outcome dict into two arm-level row dicts."""
    name = (outcome.get("outcome_name") or "").strip()
    arm_a = (outcome.get("arm_a") or "").strip()
    arm_b = (outcome.get("arm_b") or "").strip()

    _warn_unknown_arms(name, arm_a, arm_b, known_arms)

    base = {"Study": study_id, "Outcome": name, "Data type": "Arm level"}
    return [
        {
            **base,
            "Arm": arm_a,
            "Sample size": str(outcome.get("sample_size_a") or ""),
            "Mean": str(outcome.get("mean_a") or ""),
            "SD": str(outcome.get("sd_a") or ""),
            "SE": str(outcome.get("se_a") or ""),
            "Variance": str(outcome.get("variance_a") or ""),
            "CI level": str(outcome.get("ci_level_a") or ""),
            "CI start": str(outcome.get("ci_start_a") or ""),
            "CI end": str(outcome.get("ci_end_a") or ""),
            "t-test": str(outcome.get("t_test_a") or ""),
            "P value": str(outcome.get("p_value_a") or ""),
            "Footnotes": str(outcome.get("footnotes") or ""),
        },
        {
            **base,
            "Arm": arm_b,
            "Sample size": str(outcome.get("sample_size_b") or ""),
            "Mean": str(outcome.get("mean_b") or ""),
            "SD": str(outcome.get("sd_b") or ""),
            "SE": str(outcome.get("se_b") or ""),
            "Variance": str(outcome.get("variance_b") or ""),
            "CI level": str(outcome.get("ci_level_b") or ""),
            "CI start": str(outcome.get("ci_start_b") or ""),
            "CI end": str(outcome.get("ci_end_b") or ""),
            "t-test": str(outcome.get("t_test_b") or ""),
            "P value": str(outcome.get("p_value_b") or ""),
            "Footnotes": "",
        },
    ]


def _warn_unknown_arms(
    outcome_name: str,
    arm_a: str,
    arm_b: str,
    known_arms: set[str],
) -> None:
    """Log a warning when an extracted arm name is not in the declared interventions."""
    if known_arms:
        for arm in (arm_a, arm_b):
            if arm and arm not in known_arms:
                logger.warning(
                    f"Outcome '{outcome_name}': arm '{arm}' does not match any "
                    f"declared intervention arm ({sorted(known_arms)}). "
                    "Check the extraction output for possible mismatch."
                )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def postprocess_cochrane_outcomes(
    json_path: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    """Read a Cochrane extraction JSON and write a split-row outcomes CSV.

    Each dichotomous and continuous outcome in the JSON is expanded into
    two rows (one per arm).  Arm names in the output are validated against
    the ``interventions`` list in the JSON and a warning is logged when
    they do not match.

    Parameters
    ----------
    json_path:
        Path to the JSON file produced by ``CochraneRCTExtractionPipeline``.
    output_csv:
        Destination CSV path.  Defaults to ``<json_stem>_outcomes.csv``
        next to the JSON file.

    Returns
    -------
    Path
        The path of the written CSV file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    study_dict: dict = json.loads(json_path.read_text(encoding="utf-8"))

    chars = study_dict.get("study_characteristics") or {}

    # Derive study ID from the extracted 'study' field (STD-author-year format).
    # Fall back to id_doi, then to the JSON file stem.
    study_id: str = (
        (chars.get("study") or "").strip()
        or (chars.get("id_doi") or "").strip()
        or json_path.stem
    )

    known_arms = _known_arm_names(study_dict)

    rows: list[dict[str, str]] = []

    for outcome in study_dict.get("dichotomous_outcomes", []):
        rows.extend(_split_dichotomous(outcome, study_id, known_arms))

    for outcome in study_dict.get("continuous_outcomes", []):
        rows.extend(_split_continuous(outcome, study_id, known_arms))

    if output_csv is None:
        output_csv = json_path.with_name(f"{json_path.stem}_outcomes.csv")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_FIELDNAMES,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"Post-processed {len(rows)} outcome rows "
        f"({len(study_dict.get('dichotomous_outcomes', []))} dichotomous, "
        f"{len(study_dict.get('continuous_outcomes', []))} continuous) "
        f"→ {output_csv}"
    )
    return output_csv


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process a Cochrane extraction JSON: split outcomes into "
            "per-arm rows and write a Study+results-compatible CSV."
        )
    )
    parser.add_argument("json_path", help="Path to the extraction JSON file.")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default=None,
        help=(
            "Destination CSV path. Defaults to <json_stem>_outcomes.csv "
            "next to the JSON file."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = postprocess_cochrane_outcomes(args.json_path, args.output_csv)
    print(out)


if __name__ == "__main__":
    main()
