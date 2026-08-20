"""Generate a hierarchical prompt CSV and run runtime-dynamic DSPy extraction."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

from deet.hierarchical_mvp import AnimalRCTmodel as animal_models
from deet.hierarchical_mvp import ClimateCarbonPricingmodel as climate_carbon_pricing_models
from deet.hierarchical_mvp import CochraneRCTmodel as cochrane_models
from deet.hierarchical_mvp import ObesityRCTmodel as obesity_models
from deet.hierarchical_mvp import PrognosticModel as prognostic_models
from deet.hierarchical_mvp import RCTmodel as hierarchical_models
from deet.hierarchical_mvp.utils import (
    _open_csv_for_write,
    _write_tables,
    configure_lm,
    load_study_context,
)
from deet.logger import logger
from deet.processors.parser import parse_folder_to_markdown

DEFAULT_PROMPT_CSV_FILENAME = "hierarchical_prompts.csv"
DEFAULT_CONFIG_FILENAME = "hierarchical_config.json"
DEFAULT_BATCH_CONFIG_FILENAME = "batch_config.json"

# Each study "shape" (RCT-style/Prognostic-style/ClimateCarbonPricing-style/...) declares
# its own set of class names required to build its dynamic runtime pipeline. Class names
# ARE allowed to overlap across shapes (e.g. "Study", "Intervention") because a single CSV
# schema only ever describes one study type at a time, so each shape's resolver only ever
# looks at its own set below — never the global union. When adding a new study type/shape,
# add its own <SHAPE>_DYNAMIC_CLASSES set here and it will automatically be included in
# TARGET_DYNAMIC_CLASSES.
RCT_DYNAMIC_CLASSES = {
    "Continuous_Outcome",
    "Dichotomous_Outcome",
    "Intervention",
    "Other_Outcome",
    "Study",
    "Study_Characteristics",
}

PROGNOSTIC_DYNAMIC_CLASSES = {
    "HazardRatioOutcome",
    "OtherPrognosticOutcome",
    "PrognosticFactor",
    "PrognosticStudy",
    "PrognosticStudy_Characteristics",
}

CLIMATE_CARBON_PRICING_DYNAMIC_CLASSES = {
    "Study_Characteristics",
    "Intervention",
    "Effect_Outcome",
    "Study",
}

# Union of every shape's class names — used ONLY to decide which schema classes are
# eligible for dynamic-model generation (see `_build_dynamic_models_from_schema`). This is
# safe to keep as a flat union because that function silently skips any schema class not in
# this set; it never raises when a shape-specific class is absent, unlike the per-shape
# `_ensure_<shape>_runtime_models` resolvers, which MUST use their own dedicated set.
TARGET_DYNAMIC_CLASSES = (
    RCT_DYNAMIC_CLASSES | PROGNOSTIC_DYNAMIC_CLASSES | CLIMATE_CARBON_PRICING_DYNAMIC_CLASSES
)


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


def build_cochrane_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in CochraneRCT models."""
    rows: list[dict[str, str]] = []

    for _, cls in cochrane_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != cochrane_models.__name__:
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


def build_prognostic_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in PrognosticModel."""
    rows: list[dict[str, str]] = []

    for _, cls in prognostic_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != prognostic_models.__name__:
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


def build_obesity_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in ObesityRCT models."""
    rows: list[dict[str, str]] = []

    for _, cls in obesity_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != obesity_models.__name__:
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


def build_animal_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in AnimalRCT models."""
    rows: list[dict[str, str]] = []

    for _, cls in animal_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != animal_models.__name__:
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


def build_climate_carbon_pricing_hierarchical_prompt_rows() -> list[dict[str, str]]:
    """Build prompt rows from classes defined in ClimateCarbonPricing models."""
    rows: list[dict[str, str]] = []

    for _, cls in climate_carbon_pricing_models.__dict__.items():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != climate_carbon_pricing_models.__name__:
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
        case "CochraneRCT":
            rows = build_cochrane_hierarchical_prompt_rows()
        case "PrognosticStudy":
            rows = build_prognostic_hierarchical_prompt_rows()
        case "ObesityRCT":
            rows = build_obesity_hierarchical_prompt_rows()
        case "AnimalRCT":
            rows = build_animal_hierarchical_prompt_rows()
        case "ClimateCarbonPricing":
            rows = build_climate_carbon_pricing_hierarchical_prompt_rows()
        case _:
            raise ValueError(
                f"Unsupported study_type '{study_type}'. Supported: RCT, CochraneRCT, PrognosticStudy, ObesityRCT, AnimalRCT, ClimateCarbonPricing"
            )

    if csv_outpath is None:
        output_csv_path = Path.cwd() / DEFAULT_PROMPT_CSV_FILENAME
    else:
        output_csv_path = Path(csv_outpath)
        if not output_csv_path.is_absolute():
            output_csv_path = Path.cwd() / output_csv_path

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_csv_for_write(output_csv_path) as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["class", "attribute", "prompt", "datatype"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Hierarchical prompts CSV saved to {output_csv_path}")
    return output_csv_path


def _resolve_dtype(
    datatype: str,
    schema: dict[str, list[dict[str, str]]] | None = None,
    nested_cache: dict[str, type[BaseModel]] | None = None,
) -> Any:
    normalized = datatype.strip().lower()

    # If the datatype names another class that itself has rows in the CSV schema
    # (e.g. a nested/enum-like class like InterventionType), build it dynamically
    # from those rows so edits to its field prompts take effect, instead of always
    # falling back to a hardcoded/static class or a plain str.
    if schema is not None and nested_cache is not None:
        for class_name in schema:
            if class_name.lower() != normalized:
                continue
            if class_name not in nested_cache:
                definitions: dict[str, tuple[Any, Field]] = {}
                for field_def in schema[class_name]:
                    inner_dtype = _resolve_dtype(field_def["datatype"], schema, nested_cache)
                    definitions[field_def["attribute"]] = (
                        inner_dtype,
                        Field(default="", description=field_def["prompt"]),
                    )
                nested_cache[class_name] = create_model(
                    f"Dynamic{class_name}",
                    __base__=BaseModel,
                    **definitions,
                )
            return nested_cache[class_name]

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

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csvfile:
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
    nested_cache: dict[str, type[BaseModel]] = {}

    for class_name, fields in schema.items():
        if class_name not in TARGET_DYNAMIC_CLASSES:
            continue

        definitions: dict[str, tuple[Any, Field]] = {}
        for field_def in fields:
            dtype = _resolve_dtype(field_def["datatype"], schema, nested_cache)
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


def _ensure_rct_runtime_models(
    schema: dict[str, list[dict[str, str]]],
    dynamic_models: dict[str, type[BaseModel]],
) -> dict[str, type[BaseModel]]:
    runtime_models: dict[str, type[BaseModel]] = {}

    for class_name in RCT_DYNAMIC_CLASSES:
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
            "interventions": dspy.InputField(
                desc="Interventions identified in step 1."
            ),
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
            'If a value is not reported, use the string "NR".'
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
            "interventions": dspy.InputField(
                desc="Interventions identified in step 1."
            ),
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
            'If a value is not reported, use the string "NR".'
        ),
    )

    extract_other_sig = _build_dynamic_signature(
        "DynamicExtractOtherOutcomes",
        {
            "context": str,
            "interventions": list[runtime_models["Intervention"]],
            "dichotomous_outcomes": list[runtime_models["Dichotomous_Outcome"]],
            "continuous_outcomes": list[runtime_models["Continuous_Outcome"]],
            "flexible_outcomes": list[runtime_models["Other_Outcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one RCT."),
            "interventions": dspy.InputField(
                desc="Interventions identified in step 1."
            ),
            "dichotomous_outcomes": dspy.InputField(
                desc="All already extracted data related to dichotomous outcomes reported in the study."
            ),
            "continuous_outcomes": dspy.InputField(
                desc="All already extracted data related to continuous outcomes reported in the study."
            ),
            "flexible_outcomes": dspy.OutputField(
                desc="All non-dichotomous, non-continuous outcomes."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same RCT context and the already-identified intervention groups,\n"
            "extract ALL other (non-dichotomous, non-continuous) outcome data reported in the text. "
            "Do not re-extract a dichotomous or continuous outcome unless you can identify new data "
            "for it that wasn't extracted in the previous steps.\n\n"
            "For EVERY other outcome found, attempt to extract the attributes that are part of the schema attached to this class.\n\n"
            "Report values exactly as they appear in the source — do not calculate or impute.\n"
            'If a value is not reported, use the string "NR".'
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
                dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
                continuous_outcomes=cont_pred.continuous_outcomes,
            )

            return study_model(
                study_characteristics=study_pred.study_characteristics,
                interventions=study_pred.interventions,
                dichotomous_outcomes=dichot_pred.dichotomous_outcomes,
                continuous_outcomes=cont_pred.continuous_outcomes,
                other_outcomes=other_pred.flexible_outcomes,
            )

    return DynamicRCTExtractionPipeline


def _validate_export_flags(config: dict[str, Any]) -> None:
    """Default and validate export_csv/export_xlsx/export_json, mirroring main_hierarchical.py."""
    config.setdefault("export_csv", True)
    config.setdefault("export_xlsx", False)
    config.setdefault("export_json", False)
    for export_key in ("export_csv", "export_xlsx", "export_json"):
        if not isinstance(config[export_key], bool):
            raise TypeError(f"Config key '{export_key}' must be a boolean.")


def _load_runtime_config(config_path: Path) -> dict[str, Any]:
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

    _validate_export_flags(config)

    return config


def _load_runtime_batch_config(config_path: Path) -> dict[str, Any]:
    """Load batch config: same shape as `_load_runtime_config` but with `input_folder`."""
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

    _validate_export_flags(config)

    return config


def _serialize_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        if hasattr(value, "value"):
            return value.value
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


def _flatten_row_for_export(row: dict[str, Any]) -> dict[str, Any]:
    """Expand nested dict values into their own columns; JSON-encode lists.

    Ensures no study_type ever writes a dict into a single CSV/XLSX cell.
    """
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                column = sub_key if sub_key not in flattened else f"{key}_{sub_key}"
                flattened[column] = sub_value
        elif isinstance(value, list):
            flattened[key] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[key] = value
    return flattened


def _fieldnames_union(rows: list[dict[str, Any]]) -> list[str]:
    """Ordered union of keys across rows, capturing per-row flattened dict columns."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _ensure_prognostic_runtime_models(
    schema: dict[str, list[dict[str, str]]],
    dynamic_models: dict[str, type[BaseModel]],
) -> dict[str, type[BaseModel]]:
    runtime_models: dict[str, type[BaseModel]] = {}

    for class_name in PROGNOSTIC_DYNAMIC_CLASSES:
        model_cls = dynamic_models.get(class_name)
        if model_cls is not None:
            runtime_models[class_name] = model_cls
            continue

        fallback = getattr(prognostic_models, class_name, None)
        if isinstance(fallback, type) and issubclass(fallback, BaseModel):
            runtime_models[class_name] = fallback
        else:
            raise ValueError(
                f"Class '{class_name}' is required by the pipeline but is missing."
            )

    if "PrognosticStudy" in schema:
        runtime_models["PrognosticStudy"] = create_model(
            "DynamicPrognosticStudy",
            __base__=BaseModel,
            study_characteristics=(
                runtime_models["PrognosticStudy_Characteristics"],
                Field(description="Study-level metadata."),
            ),
            prognostic_factors=(
                list[runtime_models["PrognosticFactor"]],
                Field(description="Prognostic factors examined in the study."),
            ),
            hazard_ratio_outcomes=(
                list[runtime_models["HazardRatioOutcome"]],
                Field(default_factory=list, description="Hazard ratio outcomes."),
            ),
            other_prognostic_outcomes=(
                list[runtime_models["OtherPrognosticOutcome"]],
                Field(default_factory=list, description="Other prognostic outcomes."),
            ),
        )

    return runtime_models


def _build_dynamic_prognostic_pipeline(
    runtime_models: dict[str, type[BaseModel]],
) -> type[dspy.Module]:
    extract_study_info_sig = _build_dynamic_signature(
        "DynamicExtractPrognosticStudyInfo",
        {
            "context": str,
            "study_characteristics": runtime_models["PrognosticStudy_Characteristics"],
            "prognostic_factors": list[runtime_models["PrognosticFactor"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one prognostic study."),
            "study_characteristics": dspy.OutputField(
                desc="Study-level metadata and characteristics."
            ),
            "prognostic_factors": dspy.OutputField(
                desc="All prognostic factors examined in the study."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given plain text (converted from PDFs to markdown) from one or more documents\n"
            "that all describe the SAME prognostic study, extract all study-level\n"
            "metadata and characteristics, and identify every distinct prognostic factor examined.\n\n"
            "Report only information that is explicitly stated in the context."
        ),
    )

    extract_hazard_ratio_sig = _build_dynamic_signature(
        "DynamicExtractHazardRatioOutcomes",
        {
            "context": str,
            "prognostic_factors": list[runtime_models["PrognosticFactor"]],
            "hazard_ratio_outcomes": list[runtime_models["HazardRatioOutcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one prognostic study."),
            "prognostic_factors": dspy.InputField(
                desc="Prognostic factors identified in step 1."
            ),
            "hazard_ratio_outcomes": dspy.OutputField(
                desc="All outcomes reported as hazard ratios."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same prognostic study context and the already-identified prognostic factors,\n"
            "extract ALL outcomes reported as hazard ratios.\n\n"
            "For EVERY hazard ratio outcome found, attempt to extract all attributes in the schema.\n\n"
            "Report numbers exactly as they appear in the source — do not calculate or impute.\n"
            "If a value is not reported, use the string \"NR\"."
        ),
    )

    extract_other_sig = _build_dynamic_signature(
        "DynamicExtractOtherPrognosticOutcomes",
        {
            "context": str,
            "prognostic_factors": list[runtime_models["PrognosticFactor"]],
            "other_prognostic_outcomes": list[runtime_models["OtherPrognosticOutcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one prognostic study."),
            "prognostic_factors": dspy.InputField(
                desc="Prognostic factors identified in step 1."
            ),
            "other_prognostic_outcomes": dspy.OutputField(
                desc="All non-hazard-ratio prognostic outcomes."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same prognostic study context and the already-identified prognostic factors,\n"
            "extract ALL outcomes NOT reported as hazard ratios (e.g. response rates, means, medians,\n"
            "odds ratios, percentages).\n\n"
            "For EVERY other prognostic outcome found, attempt to extract all attributes in the schema.\n\n"
            "Report values exactly as they appear in the source — do not calculate or impute.\n"
            "If a value is not reported, use the string \"NR\"."
        ),
    )

    study_model = runtime_models["PrognosticStudy"]

    class DynamicPrognosticExtractionPipeline(dspy.Module):
        """Dynamic runtime variant of prognostic study extraction pipeline."""

        def __init__(self) -> None:
            super().__init__()
            self.extract_study_info = dspy.Predict(extract_study_info_sig)
            self.extract_hazard_ratio = dspy.Predict(extract_hazard_ratio_sig)
            self.extract_other = dspy.Predict(extract_other_sig)

        def forward(self, context: str) -> BaseModel:
            study_pred = self.extract_study_info(context=context)
            hr_pred = self.extract_hazard_ratio(
                context=context,
                prognostic_factors=study_pred.prognostic_factors,
            )
            other_pred = self.extract_other(
                context=context,
                prognostic_factors=study_pred.prognostic_factors,
            )
            return study_model(
                study_characteristics=study_pred.study_characteristics,
                prognostic_factors=study_pred.prognostic_factors,
                hazard_ratio_outcomes=hr_pred.hazard_ratio_outcomes,
                other_prognostic_outcomes=other_pred.other_prognostic_outcomes,
            )

    return DynamicPrognosticExtractionPipeline


def _ensure_climate_carbon_pricing_runtime_models(
    schema: dict[str, list[dict[str, str]]],
    dynamic_models: dict[str, type[BaseModel]],
) -> dict[str, type[BaseModel]]:
    runtime_models: dict[str, type[BaseModel]] = {}

    for class_name in CLIMATE_CARBON_PRICING_DYNAMIC_CLASSES:
        model_cls = dynamic_models.get(class_name)
        if model_cls is not None:
            runtime_models[class_name] = model_cls
            continue

        fallback = getattr(climate_carbon_pricing_models, class_name, None)
        if isinstance(fallback, type) and issubclass(fallback, BaseModel):
            runtime_models[class_name] = fallback
        else:
            raise ValueError(
                f"Class '{class_name}' is required by the pipeline but is missing."
            )

    if "Study" in schema:
        runtime_models["Study"] = create_model(
            "DynamicClimateCarbonPricingStudy",
            __base__=BaseModel,
            study_characteristics=(
                runtime_models["Study_Characteristics"],
                Field(description="Study-level metadata."),
            ),
            interventions=(
                list[runtime_models["Intervention"]],
                Field(description="Carbon pricing interventions identified in the study."),
            ),
            effect_outcomes=(
                list[runtime_models["Effect_Outcome"]],
                Field(default_factory=list, description="Effect/outcome data."),
            ),
        )

    return runtime_models


def _build_dynamic_climate_carbon_pricing_pipeline(
    runtime_models: dict[str, type[BaseModel]],
) -> type[dspy.Module]:
    extract_study_info_sig = _build_dynamic_signature(
        "DynamicExtractClimateCarbonPricingStudyInfo",
        {
            "context": str,
            "study_characteristics": runtime_models["Study_Characteristics"],
            "interventions": list[runtime_models["Intervention"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one carbon pricing study."),
            "study_characteristics": dspy.OutputField(
                desc="Study-level metadata and characteristics."
            ),
            "interventions": dspy.OutputField(
                desc="All carbon pricing interventions identified in the study."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given plain text (converted from PDFs to markdown) from one or more documents\n"
            "that all describe the SAME study on carbon pricing, extract all study-level\n"
            "metadata and characteristics, and identify every distinct carbon pricing\n"
            "intervention analysed.\n\n"
            "Report only information that is explicitly stated in the context."
        ),
    )

    extract_effect_outcomes_sig = _build_dynamic_signature(
        "DynamicExtractEffectOutcomes",
        {
            "context": str,
            "interventions": list[runtime_models["Intervention"]],
            "effect_outcomes": list[runtime_models["Effect_Outcome"]],
        },
        {
            "context": dspy.InputField(desc="Concatenated markdown text for one carbon pricing study."),
            "interventions": dspy.InputField(
                desc="Carbon pricing interventions identified in step 1."
            ),
            "effect_outcomes": dspy.OutputField(
                desc="All effect/outcome data reported in the study."
            ),
        },
        (
            "You are a systematic review assistant.\n\n"
            "Given the same carbon pricing study context and the already-identified\n"
            "interventions, extract ALL effect/outcome data reported in the text describing\n"
            "the estimated effect of carbon pricing on emissions.\n\n"
            "For EVERY effect outcome found, attempt to extract the attributes that are part of the schema attached to this class.\n\n"
            "Report numbers exactly as they appear in the source — do not calculate or impute.\n"
            'If a value is not reported, use the string "NR".'
        ),
    )

    study_model = runtime_models["Study"]

    class DynamicClimateCarbonPricingExtractionPipeline(dspy.Module):
        """Dynamic runtime variant of ClimateCarbonPricing extraction pipeline."""

        def __init__(self) -> None:
            super().__init__()
            self.extract_study_info = dspy.Predict(extract_study_info_sig)
            self.extract_effect_outcomes = dspy.Predict(extract_effect_outcomes_sig)

        def forward(self, context: str) -> BaseModel:
            study_pred = self.extract_study_info(context=context)
            effect_pred = self.extract_effect_outcomes(
                context=context,
                interventions=study_pred.interventions,
            )
            return study_model(
                study_characteristics=study_pred.study_characteristics,
                interventions=study_pred.interventions,
                effect_outcomes=effect_pred.effect_outcomes,
            )

    return DynamicClimateCarbonPricingExtractionPipeline


def _build_dynamic_pipeline_for_study_type(
    study_type: str,
    schema: dict[str, list[dict[str, str]]],
) -> type[dspy.Module]:
    """Build the dynamic DSPy pipeline class for a study type from a loaded CSV schema."""
    dynamic_models = _build_dynamic_models_from_schema(schema)

    if study_type == "PrognosticStudy":
        runtime_models = _ensure_prognostic_runtime_models(schema, dynamic_models)
        return _build_dynamic_prognostic_pipeline(runtime_models)

    if study_type == "ClimateCarbonPricing":
        runtime_models = _ensure_climate_carbon_pricing_runtime_models(schema, dynamic_models)
        return _build_dynamic_climate_carbon_pricing_pipeline(runtime_models)

    runtime_models = _ensure_rct_runtime_models(schema, dynamic_models)
    return _build_dynamic_rct_pipeline(runtime_models)


def _write_dynamic_study_outputs(
    study: Any,
    schema: dict[str, list[dict[str, str]]],
    study_type: str,
    output_parent_dir: Path,
    study_name: str,
    timestamp: str,
    suffix: str,
    *,
    export_csv_files: bool = True,
    export_xlsx_file: bool = False,
    export_json_file: bool = False,
    flat_output: bool = False,
) -> Path:
    """Write dynamic extraction outputs (JSON/CSV/XLSX) for one study, honoring export flags.

    When `flat_output` is True (used by the batch command), CSV/XLSX files are written
    directly into `output_parent_dir` with `study_name` embedded in each filename,
    instead of nested under `output_parent_dir/<study_name>/`.
    """
    csv_dir = output_parent_dir if flat_output else output_parent_dir / study_name

    # Falls back to output_parent_dir when export_json_file is False, so callers
    # always get a valid path back regardless of which export flags are set.
    output_path = output_parent_dir

    if study_type == "PrognosticStudy":
        sc_schema = schema.get("PrognosticStudy_Characteristics", [])
        pf_schema = schema.get("PrognosticFactor", [])
        hr_schema = schema.get("HazardRatioOutcome", [])
        op_schema = schema.get("OtherPrognosticOutcome", [])

        study_row = _project_instance_to_schema(study.study_characteristics, sc_schema)
        pf_rows = [
            _project_instance_to_schema(item, pf_schema)
            for item in study.prognostic_factors
        ]
        hr_rows = [
            _project_instance_to_schema(item, hr_schema)
            for item in study.hazard_ratio_outcomes
        ]
        op_rows = [
            _project_instance_to_schema(item, op_schema)
            for item in study.other_prognostic_outcomes
        ]

        if export_json_file:
            dynamic_payload = {
                "study_characteristics": study_row,
                "prognostic_factors": pf_rows,
                "hazard_ratio_outcomes": hr_rows,
                "other_prognostic_outcomes": op_rows,
            }
            output_path = output_parent_dir / f"{study_name}_{timestamp}{suffix}.json"
            output_path.write_text(json.dumps(dynamic_payload, indent=2), encoding="utf-8")

        if export_csv_files or export_xlsx_file:
            raw_outcome_fieldnames: list[str] = []
            for group in (hr_schema, op_schema):
                for item in group:
                    attr = item["attribute"]
                    if attr not in raw_outcome_fieldnames:
                        raw_outcome_fieldnames.append(attr)
            # Tag each outcome row with its subtype so it survives into a single
            # combined "outcomes" table, mirroring main_hierarchical.py's output.
            combined_outcomes = [
                _flatten_row_for_export({"outcome_type": "hazard_ratio", **row})
                for row in hr_rows
            ] + [_flatten_row_for_export({"outcome_type": "other", **row}) for row in op_rows]

            flat_study_row = _flatten_row_for_export(study_row)
            flat_pf_rows = [_flatten_row_for_export(row) for row in pf_rows]

            tables: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
            if sc_schema:
                tables["study"] = (list(flat_study_row.keys()), [flat_study_row])
            if pf_schema:
                fieldnames = _fieldnames_union(flat_pf_rows) or [
                    item["attribute"] for item in pf_schema
                ]
                tables["prognostic_factors"] = (fieldnames, flat_pf_rows)
            if raw_outcome_fieldnames:
                fieldnames = _fieldnames_union(combined_outcomes) or [
                    "outcome_type",
                    *raw_outcome_fieldnames,
                ]
                tables["outcomes"] = (fieldnames, combined_outcomes)

            if tables:
                csv_dir.mkdir(parents=True, exist_ok=True)
                _write_tables(
                    tables,
                    csv_dir,
                    timestamp,
                    suffix,
                    write_csv=export_csv_files,
                    write_xlsx=export_xlsx_file,
                    study_name=study_name if flat_output else None,
                )
    elif study_type == "ClimateCarbonPricing":
        sc_schema = schema.get("Study_Characteristics", [])
        iv_schema = schema.get("Intervention", [])
        eo_schema = schema.get("Effect_Outcome", [])

        study_row = _project_instance_to_schema(study.study_characteristics, sc_schema)
        intervention_rows = [
            _project_instance_to_schema(item, iv_schema) for item in study.interventions
        ]
        effect_outcome_rows = [
            _project_instance_to_schema(item, eo_schema)
            for item in study.effect_outcomes
        ]

        if export_json_file:
            dynamic_payload = {
                "study_characteristics": study_row,
                "interventions": intervention_rows,
                "effect_outcomes": effect_outcome_rows,
            }
            output_path = output_parent_dir / f"{study_name}_{timestamp}{suffix}.json"
            output_path.write_text(json.dumps(dynamic_payload, indent=2), encoding="utf-8")

        if export_csv_files or export_xlsx_file:
            # Only one outcome subtype exists for this study shape, so no
            # outcome_type column is added (unlike RCT-style/Prognostic-style).
            flat_study_row = _flatten_row_for_export(study_row)
            flat_intervention_rows = [_flatten_row_for_export(row) for row in intervention_rows]
            flat_effect_outcome_rows = [
                _flatten_row_for_export(row) for row in effect_outcome_rows
            ]

            tables = {}
            if sc_schema:
                tables["study"] = (list(flat_study_row.keys()), [flat_study_row])
            if iv_schema:
                fieldnames = _fieldnames_union(flat_intervention_rows) or [
                    item["attribute"] for item in iv_schema
                ]
                tables["interventions"] = (fieldnames, flat_intervention_rows)
            if eo_schema:
                fieldnames = _fieldnames_union(flat_effect_outcome_rows) or [
                    item["attribute"] for item in eo_schema
                ]
                tables["outcomes"] = (fieldnames, flat_effect_outcome_rows)

            if tables:
                csv_dir.mkdir(parents=True, exist_ok=True)
                _write_tables(
                    tables,
                    csv_dir,
                    timestamp,
                    suffix,
                    write_csv=export_csv_files,
                    write_xlsx=export_xlsx_file,
                    study_name=study_name if flat_output else None,
                )
    else:
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

        if export_json_file:
            dynamic_payload = {
                "study_characteristics": study_row,
                "interventions": intervention_rows,
                "dichotomous_outcomes": dichot_rows,
                "continuous_outcomes": cont_rows,
                "other_outcomes": other_rows,
            }
            output_path = output_parent_dir / f"{study_name}_{timestamp}{suffix}.json"
            output_path.write_text(json.dumps(dynamic_payload, indent=2), encoding="utf-8")

        if export_csv_files or export_xlsx_file:
            raw_outcome_fieldnames = []
            for group in (do_schema, co_schema, oo_schema):
                for item in group:
                    attr = item["attribute"]
                    if attr not in raw_outcome_fieldnames:
                        raw_outcome_fieldnames.append(attr)
            combined_outcomes = (
                [
                    _flatten_row_for_export({"outcome_type": "dichotomous", **row})
                    for row in dichot_rows
                ]
                + [
                    _flatten_row_for_export({"outcome_type": "continuous", **row})
                    for row in cont_rows
                ]
                + [
                    _flatten_row_for_export({"outcome_type": "other", **row})
                    for row in other_rows
                ]
            )

            flat_study_row = _flatten_row_for_export(study_row)
            flat_intervention_rows = [_flatten_row_for_export(row) for row in intervention_rows]

            tables = {}
            if sc_schema:
                tables["study"] = (list(flat_study_row.keys()), [flat_study_row])
            if iv_schema:
                fieldnames = _fieldnames_union(flat_intervention_rows) or [
                    item["attribute"] for item in iv_schema
                ]
                tables["interventions"] = (fieldnames, flat_intervention_rows)
            if raw_outcome_fieldnames:
                fieldnames = _fieldnames_union(combined_outcomes) or [
                    "outcome_type",
                    *raw_outcome_fieldnames,
                ]
                tables["outcomes"] = (fieldnames, combined_outcomes)

            if tables:
                csv_dir.mkdir(parents=True, exist_ok=True)
                _write_tables(
                    tables,
                    csv_dir,
                    timestamp,
                    suffix,
                    write_csv=export_csv_files,
                    write_xlsx=export_xlsx_file,
                    study_name=study_name if flat_output else None,
                )

    return output_path


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
    model_suffix = config["llm_model"].rsplit("/", 1)[-1]
    configure_lm(
        config["llm_model"], int(config["max_tokens"]), cache=bool(config["dspy_cache"])
    )

    context = load_study_context(input_paths)

    schema = _load_prompt_schema(schema_path)
    study_type = config["study_type"]
    dynamic_pipeline_cls = _build_dynamic_pipeline_for_study_type(study_type, schema)

    study = dynamic_pipeline_cls()(context=context)

    study_name = Path(input_paths[0]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{model_suffix}" if model_suffix else ""

    json_path = _write_dynamic_study_outputs(
        study,
        schema,
        study_type,
        output_parent_dir,
        study_name,
        timestamp,
        suffix,
        export_csv_files=config["export_csv"],
        export_xlsx_file=config["export_xlsx"],
        export_json_file=config["export_json"],
    )

    logger.info(
        f"Dynamic hierarchical extraction complete. Outputs saved to {output_parent_dir}"
    )
    return json_path


def run_dynamic_batch_extraction_from_csv_schema(
    csv_path: str | Path,
    batch_config_path: str | Path | None = None,
) -> list[Path]:
    """Run dynamic CSV-schema extraction for every markdown file in a batch config's input_folder."""
    schema_path = Path(csv_path)
    if not schema_path.is_absolute():
        schema_path = Path.cwd() / schema_path
    if not schema_path.exists():
        raise FileNotFoundError(f"CSV schema not found: {schema_path}")

    if batch_config_path is None:
        cfg_path = Path.cwd() / DEFAULT_BATCH_CONFIG_FILENAME
    else:
        cfg_path = Path(batch_config_path)
        if not cfg_path.is_absolute():
            cfg_path = Path.cwd() / cfg_path

    config = _load_runtime_batch_config(cfg_path)

    input_folder = Path(config["input_folder"])
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_folder}")

    output_parent_dir = Path(config["output_parent_dir"])
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    md_paths = sorted(
        p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() == ".md"
    )
    if not md_paths:
        logger.warning(f"No markdown files found directly in {input_folder}.")
        return []

    load_dotenv()
    model_suffix = config["llm_model"].rsplit("/", 1)[-1]
    configure_lm(
        config["llm_model"], int(config["max_tokens"]), cache=bool(config["dspy_cache"])
    )

    schema = _load_prompt_schema(schema_path)
    study_type = config["study_type"]
    dynamic_pipeline_cls = _build_dynamic_pipeline_for_study_type(study_type, schema)

    logger.info(f"Found {len(md_paths)} markdown file(s) to process in {input_folder}.")
    json_paths: list[Path] = []
    for md_path in md_paths:
        logger.info(f"--- Processing {md_path.name} ---")
        try:
            context = load_study_context([str(md_path)])
            study = dynamic_pipeline_cls()(context=context)

            study_name = md_path.stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{model_suffix}" if model_suffix else ""

            json_path = _write_dynamic_study_outputs(
                study,
                schema,
                study_type,
                output_parent_dir,
                study_name,
                timestamp,
                suffix,
                export_csv_files=config["export_csv"],
                export_xlsx_file=config["export_xlsx"],
                export_json_file=config["export_json"],
                flat_output=True,
            )
            json_paths.append(json_path)
        except Exception:
            logger.exception(f"Failed to process {md_path.name}, continuing with remaining files.")
            continue

    logger.info(
        f"Dynamic batch extraction complete. Outputs saved to {output_parent_dir}"
    )
    return json_paths


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
        help="Study type used for prompt CSV generation. Currently supports: RCT, CochraneRCT, PrognosticStudy, ObesityRCT, AnimalRCT, ClimateCarbonPricing.",
    )
    write_parser.add_argument(
        "--csv-outpath",
        default=None,
        help=(
            "Optional output path for prompt CSV. Defaults to "
            "<current working directory>/hierarchical_prompts.csv."
        ),
    )

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
        help=(
            "Run dynamic extraction from a CSV schema and JSON config for one study. "
            "Mimics main_hierarchical config input with one additional CSV argument."
        ),
    )
    single_parser.add_argument(
        "csv_path",
        help="Path to CSV schema used to build runtime dynamic models.",
    )
    single_parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_CONFIG_FILENAME,
        help=(
            "Path to JSON config with study_type, llm_model, input_paths, "
            "output_parent_dir, max_tokens, and dspy_cache. Defaults to "
            "hierarchical_config.json."
        ),
    )

    batch_parser = subparsers.add_parser(
        "predict_batch",
        help=(
            "Run dynamic extraction from a CSV schema for every markdown file in a "
            "batch config's input_folder."
        ),
    )
    batch_parser.add_argument(
        "csv_path",
        help="Path to CSV schema used to build runtime dynamic models.",
    )
    batch_parser.add_argument(
        "batch_config_path",
        nargs="?",
        default=DEFAULT_BATCH_CONFIG_FILENAME,
        help=(
            "Path to JSON config with study_type, llm_model, input_folder, "
            "output_parent_dir, max_tokens, and dspy_cache. Same shape as the "
            "single-study config, but 'input_paths' is replaced with a single "
            "'input_folder' path. Defaults to batch_config.json."
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
        case "parse_pdfs":
            created = parse_folder_to_markdown(Path(args.input_folder))
            print(f"Created {len(created)} markdown file(s).")
        case "predict_single_study":
            output_path = run_dynamic_extraction_from_csv_schema(
                csv_path=args.csv_path,
                config_path=args.config_path,
            )
            print(output_path)
        case "predict_batch":
            output_paths = run_dynamic_batch_extraction_from_csv_schema(
                csv_path=args.csv_path,
                batch_config_path=args.batch_config_path,
            )
            for output_path in output_paths:
                print(output_path)
        case _:
            raise ValueError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    main()
