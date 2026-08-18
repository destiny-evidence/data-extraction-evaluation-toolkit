"""Entry point for hierarchical RCT outcome data extraction MVP."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deet.hierarchical_mvp.AnimalRCTextraction import AnimalRCTExtractionPipeline
from deet.hierarchical_mvp.AnimalRCTmodel import Study as AnimalStudy
from deet.hierarchical_mvp.CochraneRCTextraction import CochraneRCTExtractionPipeline
from deet.hierarchical_mvp.ObesityRCTextraction import ObesityRCTExtractionPipeline
from deet.hierarchical_mvp.ObesityRCTmodel import Study as ObesityStudy
from deet.hierarchical_mvp.PrognosticExtraction import PrognosticExtractionPipeline
from deet.hierarchical_mvp.PrognosticModel import PrognosticStudy
from deet.hierarchical_mvp.RCTextraction import RCTExtractionPipeline
from deet.hierarchical_mvp.RCTmodel import Study
from deet.hierarchical_mvp.utils import (
    configure_lm,
    export_animal_csv,
    export_cochrane_csv,
    export_csv,
    export_obesity_csv,
    export_prognostic_csv,
    load_study_context,
)
from deet.logger import logger
from deet.processors.parser import parse_folder_to_markdown

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_SINK_ID: int | None = None
DEFAULT_CONFIG_FILENAME = "hierarchical_config.json"
DEFAULT_BATCH_CONFIG_FILENAME = "batch_config.json"
EXAMPLE_CONFIG_JSON = """{
    \"study_type\": \"RCT\",
    \"llm_model\": \"azure/gpt-5.2\",
    \"max_tokens\": 30000,
    \"dspy_cache\": false,
    \"input_paths\": [
        \"misc/hierarchical_mvp/input/mira_rct/main.md\"
    ],
    \"output_parent_dir\": \"misc/hierarchical_mvp/output/mira_rct\",
    \"export_csv\": true,
    \"export_xlsx\": false,
    \"export_json\": false
}"""
EXAMPLE_BATCH_CONFIG_JSON = """{
    \"study_type\": \"RCT\",
    \"llm_model\": \"azure/gpt-5.2\",
    \"max_tokens\": 30000,
    \"dspy_cache\": false,
    \"input_folder\": \"misc/hierarchical_mvp/input/mira_rct\",
    \"output_parent_dir\": \"misc/hierarchical_mvp/output/mira_rct\",
    \"export_csv\": true,
    \"export_xlsx\": false,
    \"export_json\": false
}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the hierarchical extraction CLI."""
    parser = argparse.ArgumentParser(
        description="Hierarchical extraction CLI: parse documents to markdown, "
        "or run extraction for a single study or a batch of studies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_pdfs_parser = subparsers.add_parser(
        "parse_pdfs",
        help="Convert every supported file directly inside a folder to markdown.",
    )
    parse_pdfs_parser.add_argument(
        "input_folder",
        help="Path to a folder of files to convert (non-recursive).",
    )

    single_parser = subparsers.add_parser(
        "predict_single_study",
        help="Run hierarchical extraction for one study using a JSON config file.",
    )
    single_parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help=(
            "Path to JSON config with study_type, input_paths, output_parent_dir, "
            "max_tokens, and dspy_cache. Optionally set export_csv (default true), "
            "export_xlsx, and export_json (both default false) to control outputs."
        ),
    )

    batch_parser = subparsers.add_parser(
        "predict_batch",
        help="Run hierarchical extraction for every markdown file in a folder.",
    )
    batch_parser.add_argument(
        "batch_config_path",
        nargs="?",
        default=DEFAULT_BATCH_CONFIG_FILENAME,
        help=(
            "Path to JSON config with study_type, input_folder, output_parent_dir, "
            "max_tokens, and dspy_cache. Same shape as the single-study config, but "
            "'input_paths' is replaced with a single 'input_folder' path."
        ),
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


def _validate_common_config(config: dict[str, Any]) -> None:
    """Validate config keys shared by single-study and batch configs."""
    if not isinstance(config["study_type"], str):
        raise TypeError("Config key 'study_type' must be a string.")

    if not isinstance(config["llm_model"], str):
        raise TypeError("Config key 'llm_model' must be a string.")

    if not isinstance(config["output_parent_dir"], str):
        raise TypeError("Config key 'output_parent_dir' must be a string.")

    if not isinstance(config["max_tokens"], int):
        raise TypeError("Config key 'max_tokens' must be an integer.")

    if config["max_tokens"] <= 0:
        raise ValueError("Config key 'max_tokens' must be greater than 0.")

    if not isinstance(config["dspy_cache"], bool):
        raise TypeError("Config key 'dspy_cache' must be a boolean.")

    # Optional output-format toggles: default to CSV-only when none are given.
    config.setdefault("export_csv", True)
    config.setdefault("export_xlsx", False)
    config.setdefault("export_json", False)
    for export_key in ("export_csv", "export_xlsx", "export_json"):
        if not isinstance(config[export_key], bool):
            raise TypeError(f"Config key '{export_key}' must be a boolean.")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate hierarchical extraction configuration from JSON."""
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_keys = {
        "study_type",
        "llm_model",
        "input_paths",
        "output_parent_dir",
        "max_tokens",
        "dspy_cache",
    }
    missing = required_keys.difference(config)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Config file is missing required key(s): {missing_sorted}")

    if not isinstance(config["input_paths"], list) or not all(
        isinstance(item, str) for item in config["input_paths"]
    ):
        raise TypeError("Config key 'input_paths' must be a list of strings.")

    if not config["input_paths"]:
        raise ValueError(
            "Config key 'input_paths' must contain at least one file path."
        )

    _validate_common_config(config)

    return config


def load_batch_config(config_path: Path) -> dict[str, Any]:
    """Load and validate batch hierarchical extraction configuration from JSON."""
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required_keys = {
        "study_type",
        "llm_model",
        "input_folder",
        "output_parent_dir",
        "max_tokens",
        "dspy_cache",
    }
    missing = required_keys.difference(config)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Config file is missing required key(s): {missing_sorted}")

    if not isinstance(config["input_folder"], str):
        raise TypeError("Config key 'input_folder' must be a string.")

    _validate_common_config(config)

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


def validate_create_batch_paths(config: dict[str, Any]) -> tuple[str, str]:
    """Validate the batch input folder exists and create output directory if needed."""
    input_folder = resolve_path(config["input_folder"])
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_folder}")

    output_dir = resolve_path(config["output_parent_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(input_folder), str(output_dir)


def extract(
    context: str, study_type: str
) -> Study | PrognosticStudy | ObesityStudy | AnimalStudy:
    """Run the configured extraction pipeline for a supported study type."""
    logger.info("Running extraction pipeline...")
    match study_type:
        case "RCT":
            pipeline = RCTExtractionPipeline()
            return pipeline(context=context)
        case "CochraneRCT":
            pipeline = CochraneRCTExtractionPipeline()
            return pipeline(context=context)
        case "PrognosticStudy":
            pipeline = PrognosticExtractionPipeline()
            return pipeline(context=context)
        case "ObesityRCT":
            pipeline = ObesityRCTExtractionPipeline()
            return pipeline(context=context)
        case "AnimalRCT":
            pipeline = AnimalRCTExtractionPipeline()
            return pipeline(context=context)
        case _:
            raise ValueError(
                f"Unsupported study_type '{study_type}'. Supported: RCT, CochraneRCT, PrognosticStudy, ObesityRCT, AnimalRCT"
            )


def save_data(
    study: Study | PrognosticStudy | ObesityStudy | AnimalStudy,
    input_paths: list[str],
    output_parent_dir: str,
    study_type: str = "RCT",
    model_suffix: str = "",
    export_csv_files: bool = True,
    export_xlsx_file: bool = False,
    export_json_file: bool = False,
    flat_output: bool = False,
) -> None:
    """Persist extracted study payload to JSON, CSV, and/or XLSX outputs, as requested.

    When `flat_output` is True (used by `predict_batch`), CSV/XLSX files are written
    directly into `output_parent_dir` with the study name embedded in each filename,
    instead of nested under `output_parent_dir/<study_name>/`.
    """
    output_dir = Path(output_parent_dir)

    study_name = Path(input_paths[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{model_suffix}" if model_suffix else ""

    if export_json_file:
        output_data = json.dumps(study.model_dump(), indent=2)
        logger.info("=== Extracted Study Data ===")
        logger.info(output_data)

        output_path = output_dir / f"{study_name}_{timestamp}{suffix}.json"
        output_path.write_text(output_data, encoding="utf-8")
        logger.info(f"JSON saved to {output_path}")

    if not (export_csv_files or export_xlsx_file):
        return

    match study_type:
        case "CochraneRCT":
            export_cochrane_csv(
                study,
                study_name,
                output_dir,
                timestamp,
                model_suffix,
                write_csv=export_csv_files,
                write_xlsx=export_xlsx_file,
                flat_output=flat_output,
            )
        case "PrognosticStudy":
            export_prognostic_csv(
                study,
                study_name,
                output_dir,
                timestamp,
                model_suffix,
                write_csv=export_csv_files,
                write_xlsx=export_xlsx_file,
                flat_output=flat_output,
            )
        case "ObesityRCT":
            export_obesity_csv(
                study,
                study_name,
                output_dir,
                timestamp,
                model_suffix,
                write_csv=export_csv_files,
                write_xlsx=export_xlsx_file,
                flat_output=flat_output,
            )
        case "AnimalRCT":
            export_animal_csv(
                study,
                study_name,
                output_dir,
                timestamp,
                model_suffix,
                write_csv=export_csv_files,
                write_xlsx=export_xlsx_file,
                flat_output=flat_output,
            )
        case _:
            export_csv(
                study,
                study_name,
                output_dir,
                timestamp,
                model_suffix,
                write_csv=export_csv_files,
                write_xlsx=export_xlsx_file,
                flat_output=flat_output,
            )


def run_parse_pdfs(input_folder: str) -> None:
    """CLI handler: convert every supported file in a folder to markdown."""
    folder = resolve_path(input_folder)
    try:
        created = parse_folder_to_markdown(folder)
    except NotADirectoryError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc

    if created:
        logger.info(f"Created {len(created)} markdown file(s) in {folder}.")
    else:
        logger.info(f"No new markdown files were created in {folder}.")


def run_predict_single_study(config_arg_path: str) -> None:
    """CLI handler: run hierarchical extraction for a single study from a JSON config."""
    config_arg = Path(config_arg_path)
    config_path = config_arg if config_arg.is_absolute() else Path.cwd() / config_arg

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        logger.error(f"Config file not found: {config_path}")
        logger.info(
            "Place a config file at this location or provide a path explicitly, "
            "for example: python deet/main_hierarchical.py predict_single_study <path-to-config.json>"
        )
        logger.info(
            f"Default expected filename in current directory: {DEFAULT_CONFIG_FILENAME}"
        )
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

    load_dotenv()
    model_suffix = config["llm_model"].rsplit("/", 1)[-1]
    configure_lm(config["llm_model"], config["max_tokens"], cache=config["dspy_cache"])

    context = read_concatenade_mds(input_paths)  # get text from the md inputs
    study = extract(context=context, study_type=config["study_type"])  # do extraction
    save_data(
        study=study,
        input_paths=input_paths,
        output_parent_dir=output_parent_dir,
        study_type=config["study_type"],
        model_suffix=model_suffix,
        export_csv_files=config["export_csv"],
        export_xlsx_file=config["export_xlsx"],
        export_json_file=config["export_json"],
    )  # does what it says (I hope :) )


def run_predict_batch(batch_config_arg_path: str) -> None:
    """CLI handler: run hierarchical extraction for every markdown file in a folder."""
    config_arg = Path(batch_config_arg_path)
    config_path = config_arg if config_arg.is_absolute() else Path.cwd() / config_arg

    try:
        config = load_batch_config(config_path)
    except FileNotFoundError as exc:
        logger.error(f"Batch config file not found: {config_path}")
        logger.info(
            "Place a config file at this location or provide a path explicitly, "
            "for example: python deet/main_hierarchical.py predict_batch <path-to-batch_config.json>"
        )
        logger.info(
            f"Default expected filename in current directory: {DEFAULT_BATCH_CONFIG_FILENAME}"
        )
        logger.info("Example batch config content:")
        logger.info(EXAMPLE_BATCH_CONFIG_JSON)
        raise SystemExit(1) from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error(f"Could not load batch config from {config_path}: {exc}")
        logger.info("Example batch config content:")
        logger.info(EXAMPLE_BATCH_CONFIG_JSON)
        raise SystemExit(1) from exc

    input_folder, output_parent_dir = validate_create_batch_paths(config)

    md_paths = sorted(
        p for p in Path(input_folder).iterdir() if p.is_file() and p.suffix.lower() == ".md"
    )
    if not md_paths:
        logger.warning(f"No markdown files found directly in {input_folder}.")
        return

    load_dotenv()
    model_suffix = config["llm_model"].rsplit("/", 1)[-1]
    configure_lm(config["llm_model"], config["max_tokens"], cache=config["dspy_cache"])

    logger.info(f"Found {len(md_paths)} markdown file(s) to process in {input_folder}.")
    for md_path in md_paths:
        logger.info(f"--- Processing {md_path.name} ---")
        try:
            context = read_concatenade_mds([str(md_path)])
            study = extract(context=context, study_type=config["study_type"])
            save_data(
                study=study,
                input_paths=[str(md_path)],
                output_parent_dir=output_parent_dir,
                study_type=config["study_type"],
                model_suffix=model_suffix,
                export_csv_files=config["export_csv"],
                export_xlsx_file=config["export_xlsx"],
                export_json_file=config["export_json"],
                flat_output=True,
            )
        except Exception:
            logger.exception(f"Failed to process {md_path.name}, continuing with remaining files.")
            continue


def main() -> None:
    """CLI entrypoint for the hierarchical extraction CLI."""
    setup_console_logging()
    args = parse_args()

    match args.command:
        case "parse_pdfs":
            run_parse_pdfs(args.input_folder)
        case "predict_single_study":
            run_predict_single_study(args.config_path)
        case "predict_batch":
            run_predict_batch(args.batch_config_path)


if __name__ == "__main__":
    main()
