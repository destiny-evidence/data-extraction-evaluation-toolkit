"""Entry point for hierarchical RCT outcome data extraction MVP."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deet.hierarchical_mvp.RCTextraction import RCTExtractionPipeline
from deet.hierarchical_mvp.RCTmodel import Study
from deet.hierarchical_mvp.utils import configure_lm, export_csv, load_study_context
from deet.logger import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_SINK_ID: int | None = None
DEFAULT_CONFIG_FILENAME = "hierarchical_config.json"
EXAMPLE_CONFIG_JSON = """{
    \"study_type\": \"RCT\",
    \"max_tokens\": 30000,
    \"dspy_cache\": false,
    \"input_paths\": [
        \"misc/hierarchical_mvp/input/mira_rct/main.md\"
    ],
    \"output_parent_dir\": \"misc/hierarchical_mvp/output/mira_rct\"
}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config-driven hierarchical extraction."""
    parser = argparse.ArgumentParser(
        description="Run hierarchical extraction using a JSON config file.",
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help="Path to JSON config with study_type, input_paths, output_parent_dir, max_tokens, and dspy_cache.",
    )
    return parser.parse_args()


def setup_console_logging() -> None:
    """Attach a console sink so logger messages are shown in terminal output."""
    global CONSOLE_SINK_ID
    if CONSOLE_SINK_ID is None:
        CONSOLE_SINK_ID = logger.add(sys.stdout, level="INFO", format="{message}")


def resolve_path(path_value: str) -> Path:
    """Resolve a path string against repository root when it is relative."""
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate hierarchical extraction configuration from JSON."""
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_keys = {
        "study_type",
        "input_paths",
        "output_parent_dir",
        "max_tokens",
        "dspy_cache",
    }
    missing = required_keys.difference(config)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Config file is missing required key(s): {missing_sorted}")

    if not isinstance(config["study_type"], str):
        raise TypeError("Config key 'study_type' must be a string.")

    if not isinstance(config["input_paths"], list) or not all(
        isinstance(item, str) for item in config["input_paths"]
    ):
        raise TypeError("Config key 'input_paths' must be a list of strings.")

    if not config["input_paths"]:
        raise ValueError(
            "Config key 'input_paths' must contain at least one file path."
        )

    if not isinstance(config["output_parent_dir"], str):
        raise TypeError("Config key 'output_parent_dir' must be a string.")

    if not isinstance(config["max_tokens"], int):
        raise TypeError("Config key 'max_tokens' must be an integer.")

    if config["max_tokens"] <= 0:
        raise ValueError("Config key 'max_tokens' must be greater than 0.")

    if not isinstance(config["dspy_cache"], bool):
        raise TypeError("Config key 'dspy_cache' must be a boolean.")

    return config


def read_concatenade_mds(input_paths: list[str]) -> str:
    """Read and concatenate markdown input files into a single extraction context."""
    logger.info(f"Loading context from {len(input_paths)} file(s)...")
    context = load_study_context(input_paths)
    logger.info(f"Context loaded ({len(context):,} characters).")
    return context


def validate_create_paths(config: dict[str, Any]) -> tuple[list[str], str]:
    """Validate input files exist and create output directory if needed."""
    input_paths = [str(resolve_path(path)) for path in config["input_paths"]]

    missing_files = [path for path in input_paths if not Path(path).is_file()]
    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(f"Input file(s) not found: {missing_str}")

    output_dir = resolve_path(config["output_parent_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return input_paths, str(output_dir)


def extract(context: str, study_type: str) -> Study:
    """Run the configured extraction pipeline for a supported study type."""
    logger.info("Running extraction pipeline...")
    match study_type:
        case "RCT":
            pipeline = RCTExtractionPipeline()
            return pipeline(context=context)
        case _:
            raise ValueError(f"Unsupported study_type '{study_type}'. Supported: RCT")


def save_data(study: Study, input_paths: list[str], output_parent_dir: str) -> None:
    """Persist extracted study payload to JSON and CSV outputs."""
    output_dir = Path(output_parent_dir)

    study_name = Path(input_paths[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_data = json.dumps(study.model_dump(), indent=2)
    logger.info("=== Extracted Study Data ===")
    logger.info(output_data)

    output_path = output_dir / f"{study_name}_{timestamp}.json"
    output_path.write_text(output_data, encoding="utf-8")
    logger.info(f"JSON saved to {output_path}")

    export_csv(study, study_name, output_dir, timestamp)


def main() -> None:
    """CLI entrypoint for hierarchical extraction using a JSON config file."""
    setup_console_logging()
    args = parse_args()

    config_arg_path = Path(args.config_path)
    if config_arg_path.is_absolute():
        config_path = config_arg_path
    else:
        config_path = Path.cwd() / config_arg_path

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        logger.error(f"Config file not found: {config_path}")
        logger.info(
            "Place a config file at this location or provide a path explicitly, "
            "for example: python deet/main_hierarchical.py <path-to-config.json>"
        )
        logger.info(f"Default expected filename in current directory: {DEFAULT_CONFIG_FILENAME}")
        logger.info("Example config content:")
        logger.info(EXAMPLE_CONFIG_JSON)
        raise SystemExit(1) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error(f"Could not load config from {config_path}: {exc}")
        logger.info("Example config content:")
        logger.info(EXAMPLE_CONFIG_JSON)
        raise SystemExit(1) from exc

    input_paths, output_parent_dir = validate_create_paths(
        config
    )  # I want to validate in and output paths and create output folder if needed BEFORE the extraction, to avoid running the LLM and then losing data due to unnecessary path/permission issues.

    load_dotenv()  # ideally this bit could/should also be defined in the custom config, but am taking it from the normal DEET .env for now
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise OSError("LLM_MODEL is not set. Add your Azure deployment name to .env.")
    configure_lm(model, config["max_tokens"], cache=config["dspy_cache"])

    context = read_concatenade_mds(input_paths)  # get text from the md inputs
    study = extract(context=context, study_type=config["study_type"])  # do extraction
    save_data(
        study=study,
        input_paths=input_paths,
        output_parent_dir=output_parent_dir,
    )  # does what it says (I hope :) )


if __name__ == "__main__":
    main()
