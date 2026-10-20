"""Shared classification framework and instructions for programme extraction."""

from __future__ import annotations

import csv
import warnings
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from deet.logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

FRAMEWORK_COLUMNS = (
    "category_id",
    "category_group",
    "category_statement",
    "inclusion_criteria",
    "exclusion_criteria",
)

SDG_CATEGORIES = (
    "No poverty",
    "Zero hunger",
    "Good health and well-being",
    "Quality education",
    "Gender equality",
    "Clean water and sanitation",
    "Affordable and clean energy",
    "Decent work and economic growth",
    "Industry, innovation and infrastructure",
    "Reduced inequalities",
    "Sustainable cities and communities",
    "Responsible consumption and production",
    "Climate action",
    "Life below water",
    "Life on land",
    "Peace, justice, and strong institutions",
    "Partnerships for the goals",
)

PROGRAMME_EXTRACTION_INSTRUCTIONS = """
Extract programme characteristics and every distinct intervention from the
evaluation report. Report only explicitly stated information and preserve names.
For each intervention, extract intended_results: its stated aims, targets and
ideal outcomes, regardless of actual results. Distinguish aims from actual results
and describe source-supported differences without calculating new values or
inventing aims or explanations. State when an aim, result or comparison is absent.
"""

OUTCOME_EXTRACTION_INSTRUCTIONS = """
Extract every reported outcome using the previously identified programme
characteristics and interventions. Preserve exact intervention names. Report only
source-supported results.
In baseline_data, extract reported quantitative and qualitative conditions for
this outcome before the intervention was implemented, preserving dates, values
and context. Keep baseline conditions separate from targets and actual results;
do not calculate or infer them from later changes. Leave blank if not reported.
Extract quantitative and qualitative results separately for each specific outcome,
assigning 'NR' when that type of result is not reported. In source_paragraph,
provide brief verbatim quotes supporting each quantitative and qualitative result,
label the result type and include document, section and page where available.
Never invent quotes or locations. Intended results belong to interventions,
not individual outcomes.
For SDG_categories, select all fitting SDGs based on the reported outcome evidence,
even if the report does not explicitly name an SDG. Use only the exact labels
listed in the SDG_categories field description, separated by semicolons without
numbers, prefixes or duplicates. Leave blank if none applies.
For outcome_category_classification, analytical inference is also permitted:
use the supplied classification framework and its inclusion and exclusion rules to
select every supported category. Return exact '<category_id>: <category_statement>'
entries separated by semicolons, or an empty string if none is supported.
Qualitative findings can support a match without reporting the exact indicator
percentage, but must concern the indicator's construct and relevant population.
Do not treat contextual national statistics, intended aims, recommendations or
unrelated activities as achieved outcome evidence. Preserve unmatched outcomes.
Include barriers, enablers, lessons learned, achievements and recommendations
within the corresponding outcome. Do not combine unrelated findings.
"""


def _framework_entries(framework: str) -> set[str]:
    reader = csv.DictReader(StringIO(framework), strict=True)
    if reader.fieldnames != list(FRAMEWORK_COLUMNS):
        msg = (
            "Classification framework must have these CSV columns: "
            f"{FRAMEWORK_COLUMNS}"
        )
        raise ValueError(msg)
    entries: set[str] = set()
    identifiers: set[str] = set()
    for row in reader:
        if set(row) != set(FRAMEWORK_COLUMNS) or any(
            not value or not value.strip() for value in row.values()
        ):
            msg = (
                f"Classification framework row {reader.line_num} "
                "has missing or extra data"
            )
            raise ValueError(msg)
        identifier = row["category_id"]
        if any(
            row[column] != row[column].strip()
            for column in ("category_id", "category_statement")
        ):
            msg = "Category IDs and statements cannot have surrounding whitespace"
            raise ValueError(msg)
        if identifier in identifiers:
            msg = f"Duplicate category ID: {identifier}"
            raise ValueError(msg)
        if ";" in identifier or ";" in row["category_statement"]:
            msg = "Category IDs and statements cannot contain semicolons"
            raise ValueError(msg)
        identifiers.add(identifier)
        entries.add(f"{identifier}: {row['category_statement']}")
    if not entries:
        msg = "Classification framework must contain at least one category"
        raise ValueError(msg)
    return entries


def load_indicator_framework(path: Path | None = None) -> str:
    """Load a complete CSV framework, defaulting to the supplied UNICEF table."""
    if path is None:
        path = Path(__file__).with_name("unicef_outcome_indicators.csv")
    framework = path.read_text(encoding="utf-8-sig")
    _framework_entries(framework)
    return framework


def resolve_framework_config(
    config: Mapping[str, object], config_path: Path
) -> Path | None:
    """Validate an optional framework path relative to its JSON config file."""
    if "classification_framework_path" not in config:
        return None
    value = config["classification_framework_path"]
    if not isinstance(value, str) or not value.strip():
        msg = "Config key 'classification_framework_path' must be a non-empty string"
        raise TypeError(msg)
    if config["study_type"] != "Programme":
        msg = (
            "'classification_framework_path' is supported only "
            "for study_type 'Programme'"
        )
        raise ValueError(msg)
    path = Path(value)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    load_indicator_framework(path)
    return path


def _clean_classifications(
    classifications: Iterable[str], allowed: set[str], field: str
) -> list[str]:
    cleaned = []
    for index, classification in enumerate(classifications, start=1):
        kept: list[str] = []
        removed: list[str] = []
        if classification:
            for raw_entry in classification.split(";"):
                entry = raw_entry.strip()
                if entry not in allowed or entry in kept:
                    removed.append(entry)
                else:
                    kept.append(entry)
        if removed:
            msg = (
                f"Outcome {index}: removed invalid or duplicate {field} "
                f"entries: {removed!r}"
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)
        cleaned.append("; ".join(kept))
    return cleaned


def validate_classifications(
    classifications: Iterable[str], framework: str
) -> list[str]:
    """Return valid indicator entries, warning when invalid entries are removed."""
    return _clean_classifications(
        classifications, _framework_entries(framework),
        "outcome_category_classification",
    )


def validate_sdg_categories(classifications: Iterable[str]) -> list[str]:
    """Return valid SDG labels, warning when invalid entries are removed."""
    return _clean_classifications(
        classifications, set(SDG_CATEGORIES), "SDG_categories"
    )
