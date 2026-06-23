"""Demo script for custom hierarchical extraction with hardcoded paths."""

from __future__ import annotations

from pathlib import Path

from deet.custom_hierarchical import run_dynamic_extraction_from_csv_schema


def main() -> None:
    config_path = Path("misc/hierarchical_mvp/configs/hierarchical_config.json")
    csv_path = Path("misc/hierarchical_mvp/configs/hierarchical_prompts.csv")

    output_path = run_dynamic_extraction_from_csv_schema(
        csv_path=csv_path,
        config_path=config_path,
    )
    print(f"Extraction complete. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
