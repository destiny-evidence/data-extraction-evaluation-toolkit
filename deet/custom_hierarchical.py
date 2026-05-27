"""Generate a hierarchical prompt CSV from Pydantic models."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, create_model

from deet.hierarchical_mvp.extraction import RCTExtractionPipeline
from deet.hierarchical_mvp.utils import configure_lm, load_study_context
from deet.hierarchical_mvp import models as hierarchical_models
from deet.logger import logger

OUTPUT_CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "misc"
    / "hierarchical_mvp"
    / "configs"
    / "hierarchical_prompts.csv"
)
DEFAULT_CONFIG_FILENAME = "hierarchical_config.json"


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
                    "datatype": datatype,
                    "attribute": field_name,
                    "prompt": description,
                    
                }
            )

    return rows


def write_hierarchical_prompts_csv() -> Path:
    """Write hierarchical prompt metadata to a CSV at a fixed location."""
    rows = build_hierarchical_prompt_rows()

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["class", "attribute", "prompt", "datatype"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Hierarchical prompts CSV saved to {OUTPUT_CSV_PATH}")
    return OUTPUT_CSV_PATH


def _resolve_dtype(datatype: str) -> Any:
    normalized = datatype.strip().lower()
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
        return value.model_dump()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def _project_instance_to_model(
    instance: Any,
    class_schema: list[dict[str, str]],
    dynamic_model: type[BaseModel],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_def in class_schema:
        attr = field_def["attribute"]
        payload[attr] = _serialize_value(getattr(instance, attr, ""))

    try:
        return dynamic_model(**payload).model_dump()
    except ValidationError:
        # If typed validation is too strict for extracted values, retain payload as-is.
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
    """Run extraction and project outputs dynamically based on a prompt CSV schema."""
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
    study = RCTExtractionPipeline()(context=context)

    schema = _load_prompt_schema(schema_path)
    dynamic_models = _build_dynamic_models_from_schema(schema)

    study_name = Path(input_paths[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_parent_dir / study_name
    run_dir.mkdir(parents=True, exist_ok=True)

    sc_schema = schema.get("Study_Characteristics", [])
    iv_schema = schema.get("Intervention", [])
    do_schema = schema.get("Dichotomous_Outcome", [])
    co_schema = schema.get("Continuous_Outcome", [])
    oo_schema = schema.get("Other_Outcome", [])

    study_row = _project_instance_to_model(
        study.study_characteristics,
        sc_schema,
        dynamic_models.get("Study_Characteristics", BaseModel),
    )
    intervention_rows = [
        _project_instance_to_model(
            item,
            iv_schema,
            dynamic_models.get("Intervention", BaseModel),
        )
        for item in study.interventions
    ]
    dichot_rows = [
        _project_instance_to_model(
            item,
            do_schema,
            dynamic_models.get("Dichotomous_Outcome", BaseModel),
        )
        for item in study.dichotomous_outcomes
    ]
    cont_rows = [
        _project_instance_to_model(
            item,
            co_schema,
            dynamic_models.get("Continuous_Outcome", BaseModel),
        )
        for item in study.continuous_outcomes
    ]
    other_rows = [
        _project_instance_to_model(
            item,
            oo_schema,
            dynamic_models.get("Other_Outcome", BaseModel),
        )
        for item in study.other_outcomes
    ]

    dynamic_payload = {
        "study_characteristics": study_row,
        "interventions": intervention_rows,
        "dichotomous_outcomes": dichot_rows,
        "continuous_outcomes": cont_rows,
        "other_outcomes": other_rows,
    }

    json_path = run_dir / f"{study_name}_{timestamp}.json"
    json_path.write_text(json.dumps(dynamic_payload, indent=2), encoding="utf-8")

    if sc_schema:
        _write_dict_rows_to_csv(
            run_dir / f"study_{timestamp}.csv",
            [item["attribute"] for item in sc_schema],
            [study_row],
        )
    if iv_schema:
        _write_dict_rows_to_csv(
            run_dir / f"interventions_{timestamp}.csv",
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
            run_dir / f"outcomes_{timestamp}.csv",
            outcome_fieldnames,
            combined_outcomes,
        )

    logger.info(f"Dynamic hierarchical extraction complete. Outputs saved to {run_dir}")
    return json_path


if __name__ == "__main__":
    write_hierarchical_prompts_csv()
