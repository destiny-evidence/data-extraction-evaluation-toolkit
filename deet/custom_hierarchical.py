"""Generate a hierarchical prompt CSV and run runtime-dynamic DSPy extraction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

from deet.hierarchical_mvp import RCTmodel as hierarchical_models
from deet.hierarchical_mvp.utils import configure_lm, load_study_context
from deet.logger import logger

DEFAULT_PROMPT_CSV_FILENAME = "hierarchical_prompts.csv"
DEFAULT_CONFIG_FILENAME = "hierarchical_config.json"
TARGET_DYNAMIC_CLASSES = {
    "Continuous_Outcome",
    "Dichotomous_Outcome",
    "Intervention",
    "Other_Outcome",
    "Study",
    "Study_Characteristics",
}


def build_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in hierarchical models."""
    rows: list[dict[str, str]] = []

    for _, cls in hierarchical_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != hierarchical_models.__name__:
            continue
        if not issubclass(cls, BaseModel):
            continue

        for field_name, field_info in cls.model_fields.items():
            description = field_info.description or ""
            annotation = field_info.annotation
            datatype = getattr(annotation, "__name__", str(annotation))

            rows.append(
                {
                    "class": cls.__name__,
                    "attribute": field_name,
                    "prompt": description,
                    "datatype": datatype,
                }
            )

    return rows


def write_hierarchical_prompts_csv(
    study_type: str = "RCT",
    csv_outpath: str | Path | None = None,
) -> Path:
    """Write hierarchical prompt metadata to a CSV at a fixed location."""
    match study_type:
        case "RCT":
            rows = build_hierarchical_prompt_rows()
        case _:
            raise ValueError(
                f"Unsupported study_type '{study_type}'. Supported: RCT"
            )

    if csv_outpath is None:
        output_csv_path = Path.cwd() / DEFAULT_PROMPT_CSV_FILENAME
    else:
        output_csv_path = Path(csv_outpath)
        if not output_csv_path.is_absolute():
            output_csv_path = Path.cwd() / output_csv_path

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["class", "attribute", "prompt", "datatype"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Hierarchical prompts CSV saved to {output_csv_path}")
    return output_csv_path


def _resolve_dtype(datatype: str) -> Any:
    normalized = datatype.strip().lower()

    # The prompt CSV can represent nested Pydantic model types using either
    # bare names (OutcomeTypes), prefixed names (models.OutcomeTypes), or
    # stringified annotations (<class '...OutcomeTypes'>). Normalize all forms.
    if "outcometypes" in normalized:
        return hierarchical_models.OutcomeTypes
    if "outcometimepoint" in normalized:
        return hierarchical_models.OutcomeTimePoint

    mapping: dict[str, Any] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "dict": dict[str, Any],
        "list": list[Any],
    }
    return mapping.get(normalized, str)


def _load_prompt_schema(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    schema: dict[str, list[dict[str, str]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        required = {"class", "attribute", "prompt", "datatype"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"CSV schema missing required column(s): {missing_sorted}")

        for row in reader:
            class_name = (row.get("class") or "").strip()
            attribute = (row.get("attribute") or "").strip()
            if not class_name or not attribute:
                continue

            schema.setdefault(class_name, []).append(
                {
                    "attribute": attribute,
                    "prompt": (row.get("prompt") or "").strip(),
                    "datatype": (row.get("datatype") or "str").strip(),
                }
            )

    return schema


def _build_dynamic_models_from_schema(
    schema: dict[str, list[dict[str, str]]],
) -> dict[str, type[BaseModel]]:
    dynamic_models: dict[str, type[BaseModel]] = {}

    for class_name, fields in schema.items():
        if class_name not in TARGET_DYNAMIC_CLASSES:
            continue

        definitions: dict[str, tuple[Any, Field]] = {}
        for field_def in fields:
            dtype = _resolve_dtype(field_def["datatype"])
            description = field_def["prompt"]
            definitions[field_def["attribute"]] = (
                dtype,
                Field(default="", description=description),
            )

        dynamic_models[class_name] = create_model(
            f"Dynamic{class_name}",
            __base__=BaseModel,
            **definitions,
        )

    return dynamic_models


def _ensure_runtime_models(
    schema: dict[str, list[dict[str, str]]],
    dynamic_models: dict[str, type[BaseModel]],
) -> dict[str, type[BaseModel]]:
    runtime_models: dict[str, type[BaseModel]] = {}

    for class_name in TARGET_DYNAMIC_CLASSES:
        model_cls = dynamic_models.get(class_name)
        if model_cls is not None:
            runtime_models[class_name] = model_cls
            continue

        fallback = getattr(hierarchical_models, class_name, None)
        if isinstance(fallback, type) and issubclass(fallback, BaseModel):
            runtime_models[class_name] = fallback
        else:
            raise ValueError(
                f"Class '{class_name}' is required by the pipeline but is missing."
            )

    if "Study" in schema:
        runtime_models["Study"] = create_model(
            "DynamicStudy",
            __base__=BaseModel,
            study_characteristics=(
                runtime_models["Study_Characteristics"],
                Field(description="Study-level metadata."),
            ),
            interventions=(
                list[runtime_models["Intervention"]],
                Field(description="Intervention groups in the trial."),
            ),
            dichotomous_outcomes=(
                list[runtime_models["Dichotomous_Outcome"]],
                Field(default_factory=list, description="Dichotomous outcomes."),
            ),
            continuous_outcomes=(
                list[runtime_models["Continuous_Outcome"]],
                Field(default_factory=list, description="Continuous outcomes."),
            ),
            other_outcomes=(
                list[runtime_models["Other_Outcome"]],
                Field(default_factory=list, description="Other outcomes."),
            ),
        )

    return runtime_models


def _build_dynamic_signature(
    name: str,
    annotations: dict[str, Any],
    fields: dict[str, Any],
    docstring: str,
) -> type[dspy.Signature]:
    namespace: dict[str, Any] = {"__annotations__": annotations, "__doc__": docstring}
    namespace.update(fields)
    return type(name, (dspy.Signature,), namespace)


def _build_dynamic_rct_pipeline(
    runtime_models: dict[str, type[BaseModel]],
) -> type[dspy.Module]:
    extract_study_info_sig = _build_dynamic_signature(
        "DynamicExtractStudyInfo",
        {
            "context": str,
            "study_characteristics": runtime_models["Study_Characteristics"],
            "interventions": list[runtime_models["Intervention"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one RCT."),
            "study_characteristics": dspy.OutputField(
                desc="Study-level metadata and characteristics."
            ),
            "interventions": dspy.OutputField(desc="All intervention groups in trial."),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given plain text (converted from PDFs to markdown) from one or more documents\n"
            "that all describe the SAME randomized controlled trial, extract all study-level\n"
            "metadata and characteristics, and identify every distinct intervention group (arm).\n\n"
            "Report only information that is explicitly stated in the context."
        ),
    )

    extract_dichotomous_sig = _build_dynamic_signature(
        "DynamicExtractDichotomousOutcomes",
        {
            "context": str,
            "interventions": list[runtime_models["Intervention"]],
            "dichotomous_outcomes": list[runtime_models["Dichotomous_Outcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one RCT."),
            "interventions": dspy.InputField(desc="Interventions identified in step 1."),
            "dichotomous_outcomes": dspy.OutputField(
                desc="All dichotomous outcomes reported in the study."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same RCT context and the already-identified intervention groups,\n"
            "extract ALL dichotomous (binary event) outcome data reported in the text.\n\n"
            "For EVERY dichotomous outcome found, attempt to extract the attributes that are part of the schema attached to this class.\n\n"
            "Report numbers exactly as they appear in the source — do not calculate or impute.\n"
            "If a value is not reported, use the string \"NR\"."
        ),
    )

    extract_continuous_sig = _build_dynamic_signature(
        "DynamicExtractContinuousOutcomes",
        {
            "context": str,
            "interventions": list[runtime_models["Intervention"]],
            "continuous_outcomes": list[runtime_models["Continuous_Outcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one RCT."),
            "interventions": dspy.InputField(desc="Interventions identified in step 1."),
            "continuous_outcomes": dspy.OutputField(
                desc="All continuous outcomes reported in the study."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same RCT context and the already-identified intervention groups,\n"
            "extract ALL continuous outcome data (mean ± SD) reported in the text.\n\n"
            "For EVERY continuous outcome found, attempt to extract the attributes that are part of the schema attached to this class.\n\n"
            "Report numbers exactly as they appear in the source — do not calculate or impute.\n"
            "If a value is not reported, use the string \"NR\"."
        ),
    )

    extract_other_sig = _build_dynamic_signature(
        "DynamicExtractOtherOutcomes",
        {
            "context": str,
            "interventions": list[runtime_models["Intervention"]],
            "flexible_outcomes": list[runtime_models["Other_Outcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one RCT."),
            "interventions": dspy.InputField(desc="Interventions identified in step 1."),
            "flexible_outcomes": dspy.OutputField(
                desc="All non-dichotomous, non-continuous outcomes."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same RCT context and the already-identified intervention groups,\n"
            "extract ALL other (non-dichotomous, non-continuous) outcome data reported in the text.\n\n"
            "For EVERY other outcome found, attempt to extract the attributes that are part of the schema attached to this class.\n\n"
            "Report values exactly as they appear in the source — do not calculate or impute.\n"
            "If a value is not reported, use the string \"NR\"."
        ),
    )

    study_model = runtime_models["Study"]

    class DynamicRCTExtractionPipeline(dspy.Module):
        """Dynamic runtime variant of RCT extraction pipeline."""

        def __init__(self) -> None:
            super().__init__()
            self.extract_study_info = dspy.Predict(extract_study_info_sig)
            self.extract_dichotomous = dspy.Predict(extract_dichotomous_sig)
            self.extract_continuous = dspy.Predict(extract_continuous_sig)
            self.extract_other = dspy.Predict(extract_other_sig)

        def forward(self, context: str) -> BaseModel:
            study_pred = self.extract_study_info(context=context)
            dichot_pred = self.extract_dichotomous(
                context=context,
                interventions=study_pred.interventions,
            )
            cont_pred = self.extract_continuous(
                context=context,
                interventions=study_pred.interventions,
            )
            other_pred = self.extract_other(
                context=context,
                interventions=study_pred.interventions,
            )

            return study_model(
                study_characteristics=study_pred.study_characteristics,
                interventions=study_pred.interventions,
                dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
                continuous_outcomes=cont_pred.continuous_outcomes,
                other_outcomes=other_pred.flexible_outcomes,
            )

    return DynamicRCTExtractionPipeline


def _load_runtime_config(config_path: Path) -> dict[str, Any]:
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

    return config


def _serialize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        if hasattr(value, "value"):
            return getattr(value, "value")
        value_dump = value.model_dump()
        if len(value_dump) == 1:
            return _serialize_value(next(iter(value_dump.values())))
        return value_dump
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def _project_instance_to_schema(
    instance: Any,
    class_schema: list[dict[str, str]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_def in class_schema:
        attr = field_def["attribute"]
        payload[attr] = _serialize_value(getattr(instance, attr, ""))
    return payload


def _write_dict_rows_to_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(normalized)


def run_dynamic_extraction_from_csv_schema(
    csv_path: str | Path,
    config_path: str | Path | None = None,
) -> Path:
    """Run extraction with runtime dynamic DSPy models/signatures from CSV schema."""
    schema_path = Path(csv_path)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise FileNotFoundError(f"CSV schema not found: {schema_path}")

    if config_path is None:
        cfg_path = Path.cwd() / DEFAULT_CONFIG_FILENAME
    else:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path

    config = _load_runtime_config(cfg_path)

    input_paths = [str(Path(path)) for path in config["input_paths"]]
    missing_inputs = [path for path in input_paths if not Path(path).is_file()]
    if missing_inputs:
        missing = ", ".join(missing_inputs)
        raise FileNotFoundError(f"Input file(s) not found: {missing}")

    output_parent_dir = Path(config["output_parent_dir"])
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv()
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise OSError("LLM_MODEL is not set. Add your Azure deployment name to .env.")
    configure_lm(model, int(config["max_tokens"]), cache=bool(config["dspy_cache"]))

    context = load_study_context(input_paths)

    schema = _load_prompt_schema(schema_path)
    dynamic_models = _build_dynamic_models_from_schema(schema)
    runtime_models = _ensure_runtime_models(schema, dynamic_models)
    dynamic_pipeline_cls = _build_dynamic_rct_pipeline(runtime_models)
    study = dynamic_pipeline_cls()(context=context)

    study_name = Path(input_paths[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = output_parent_dir / study_name
    csv_dir.mkdir(parents=True, exist_ok=True)

    sc_schema = schema.get("Study_Characteristics", [])
    iv_schema = schema.get("Intervention", [])
    do_schema = schema.get("Dichotomous_Outcome", [])
    co_schema = schema.get("Continuous_Outcome", [])
    oo_schema = schema.get("Other_Outcome", [])

    study_row = _project_instance_to_schema(study.study_characteristics, sc_schema)
    intervention_rows = [
        _project_instance_to_schema(item, iv_schema) for item in study.interventions
    ]
    dichot_rows = [
        _project_instance_to_schema(item, do_schema)
        for item in study.dichotomous_outcomes
    ]
    cont_rows = [
        _project_instance_to_schema(item, co_schema)
        for item in study.continuous_outcomes
    ]
    other_rows = [
        _project_instance_to_schema(item, oo_schema) for item in study.other_outcomes
    ]

    dynamic_payload = {
        "study_characteristics": study_row,
        "interventions": intervention_rows,
        "dichotomous_outcomes": dichot_rows,
        "continuous_outcomes": cont_rows,
        "other_outcomes": other_rows,
    }

    json_path = output_parent_dir / f"{study_name}_{timestamp}.json"
    json_path.write_text(json.dumps(dynamic_payload, indent=2), encoding="utf-8")

    if sc_schema:
        _write_dict_rows_to_csv(
            csv_dir / f"study_{timestamp}.csv",
            [item["attribute"] for item in sc_schema],
            [study_row],
        )
    if iv_schema:
        _write_dict_rows_to_csv(
            csv_dir / f"interventions_{timestamp}.csv",
            [item["attribute"] for item in iv_schema],
            intervention_rows,
        )

    outcome_fieldnames: list[str] = []
    for group in (do_schema, co_schema, oo_schema):
        for item in group:
            attr = item["attribute"]
            if attr not in outcome_fieldnames:
                outcome_fieldnames.append(attr)
    combined_outcomes = dichot_rows + cont_rows + other_rows
    if outcome_fieldnames:
        _write_dict_rows_to_csv(
            csv_dir / f"outcomes_{timestamp}.csv",
            outcome_fieldnames,
            combined_outcomes,
        )

    logger.info(f"Dynamic hierarchical extraction complete. Outputs saved to {output_parent_dir}")
    return json_path


def parse_custom_hierarchical_args() -> argparse.Namespace:
    """Parse CLI arguments for custom hierarchical utility commands."""
    parser = argparse.ArgumentParser(
        description="Custom hierarchical prompt generation and extraction tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser(
        "write_hierarchical_prompts_csv",
        help="Write hierarchical prompts CSV for dynamic extraction.",
    )
    write_parser.add_argument(
        "--study-type",
        default="RCT",
        help="Study type used for prompt CSV generation. Currently supports: RCT.",
    )
    write_parser.add_argument(
        "--csv-outpath",
        default=None,
        help=(
            "Optional output path for prompt CSV. Defaults to "
            "<current working directory>/hierarchical_prompts.csv."
        ),
    )

    extract_parser = subparsers.add_parser(
        "custom_extract",
        help=(
            "Run dynamic extraction from a CSV schema and JSON config. "
            "Mimics main_hierarchical config input with one additional CSV argument."
        ),
    )
    extract_parser.add_argument(
        "csv_path",
        help="Path to CSV schema used to build runtime dynamic models.",
    )
    extract_parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help=(
            "Path to JSON config with study_type, input_paths, output_parent_dir, "
            "max_tokens, and dspy_cache. Defaults to hierarchical_config.json."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the standalone custom hierarchical CLI."""
    args = parse_custom_hierarchical_args()

    match args.command:
        case "write_hierarchical_prompts_csv":
            output_path = write_hierarchical_prompts_csv(
                study_type=args.study_type,
                csv_outpath=args.csv_outpath,
            )
            print(output_path)
        case "custom_extract":
            output_path = run_dynamic_extraction_from_csv_schema(
                csv_path=args.csv_path,
                config_path=args.config_path,
            )
            print(output_path)
        case _:
            raise ValueError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    main()
