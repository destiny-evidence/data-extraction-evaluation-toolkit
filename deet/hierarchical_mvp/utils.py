from __future__ import annotations

import csv
import os
from pathlib import Path

import dspy
from deet.logger import logger

from .RCTmodel import Intervention, Study


def configure_lm(model: str, max_tokens: int, cache: bool = False) -> None:
    """Initialise DSPy with the Azure OpenAI LM."""
    api_key = os.environ.get("AZURE_API_KEY")
    api_base = os.environ.get("AZURE_API_BASE")
    if not api_key:
        raise OSError(
            "AZURE_API_KEY is not set. " "Copy .env.example to .env and add your key."
        )
    if not api_base:
        raise OSError(
            "AZURE_API_BASE is not set. "
            "Copy .env.example to .env and add your Azure endpoint."
        )
    lm = dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=max_tokens,
        cache=cache,
    )
    dspy.configure(lm=lm)


def load_study_context(markdown_paths: list[str]) -> str:
    """
    Load and concatenate plain-text markdown from multiple files.

    Files are separated by a horizontal rule so the LLM can distinguish
    document boundaries when needed.
    """
    parts: list[str] = []
    for path in markdown_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Markdown file not found: {p.resolve()}")
        parts.append(p.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def export_csv(
    study: Study,
    study_name: str,
    predictions_dir: Path,
    timestamp: str,
) -> None:
    """
        NOTE: This function probably needs to become part of the RCT pipeline
        and be called as such from the instance itself, and further downstream
        study types need to have their own export function so that we can properly use
        the input parameter and the switch statement to create Study and handle
        it well regardless of the actual study type.
        
        Write three timestamped CSV files into predictions/<study_name>/:
            - study_<timestamp>.csv
            - interventions_<timestamp>.csv
            - outcomes_<timestamp>.csv
    """
    csv_dir = predictions_dir / study_name
    csv_dir.mkdir(parents=True, exist_ok=True)

    # --- study.csv ---
    study_path = csv_dir / f"study_{timestamp}.csv"
    with study_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Study.csv_fieldnames())
        writer.writeheader()
        writer.writerow(study.to_csv_row())

    # --- interventions.csv ---
    interventions_path = csv_dir / f"interventions_{timestamp}.csv"
    with interventions_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Intervention.csv_fieldnames())
        writer.writeheader()
        for arm in study.interventions:
            writer.writerow(arm.to_csv_row())

    # --- outcomes.csv ---
    outcomes_path = csv_dir / f"outcomes_{timestamp}.csv"
    with outcomes_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=Study.outcome_csv_fieldnames(), extrasaction="ignore"
        )
        writer.writeheader()
        for outcome in (
            study.dichotomous_outcomes
            + study.continuous_outcomes
            + study.other_outcomes
        ):
            writer.writerow(outcome.to_csv_row())

    logger.info(f"CSV files saved to {csv_dir}/")
