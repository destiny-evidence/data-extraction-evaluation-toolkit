import csv
import json
from typing import get_origin
from unittest.mock import Mock, patch

import pytest
from openpyxl import load_workbook
from pydantic import BaseModel

from deet.custom_hierarchical import (
    _build_dynamic_models_from_schema,
    _build_dynamic_pipeline_for_study_type,
    _ensure_prognostic_runtime_models,
    _ensure_rct_runtime_models,
    _load_prompt_schema,
    _write_dynamic_study_outputs,
    write_hierarchical_prompts_csv,
)
from deet.hierarchical_mvp.PrognosticModel import (
    HazardRatioOutcome,
    OtherPrognosticOutcome,
    PrognosticStudy,
)
from deet.hierarchical_mvp.PrognosticModel import (
    OutcomeTimePoint as PrognosticOutcomeTimePoint,
)
from deet.hierarchical_mvp.RCTmodel import (
    Continuous_Outcome,
    Dichotomous_Outcome,
    Other_Outcome,
    OutcomeTimePoint,
    Study,
)
from deet.hierarchical_mvp.utils import export_csv, export_prognostic_csv

UNITS = ["seconds", "minutes", "hours", "days", "weeks", "months", "years", "NR"]
TIME_POINT = {
    "time_point_category": "Follow-up",
    "time_point_detail": "6-12 months after treatment",
    "time_point_value": "6-12",
    "time_point_unit": "months",
}


def _sample_model(model_cls):
    payload = {}
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is list:
            payload[name] = []
        elif get_origin(annotation) is dict:
            payload[name] = {}
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            payload[name] = _sample_model(annotation)
        else:
            payload[name] = "NR"
    return model_cls.model_validate(payload)


def _study(study_type):
    if study_type == "RCT":
        study = _sample_model(Study)
        groups = [
            (Dichotomous_Outcome, study.dichotomous_outcomes),
            (Continuous_Outcome, study.continuous_outcomes),
            (Other_Outcome, study.other_outcomes),
        ]
        time_point_cls = OutcomeTimePoint
    else:
        study = _sample_model(PrognosticStudy)
        groups = [
            (HazardRatioOutcome, study.hazard_ratio_outcomes),
            (OtherPrognosticOutcome, study.other_prognostic_outcomes),
        ]
        time_point_cls = PrognosticOutcomeTimePoint
    for model_cls, outcomes in groups:
        outcome = _sample_model(model_cls)
        outcome.outcome_time_point = time_point_cls(**TIME_POINT)
        outcomes.append(outcome)
    return study


@pytest.mark.parametrize("model_cls", [OutcomeTimePoint, PrognosticOutcomeTimePoint])
@pytest.mark.parametrize("unit", [*UNITS, "fortnights"])
def test_time_point_units(model_cls, unit):
    point = model_cls(**{**TIME_POINT, "time_point_unit": unit})
    assert point.time_point_unit == unit
    assert point.time_point_value == "6-12"
    field_schema = model_cls.model_json_schema()["properties"]["time_point_unit"]
    assert field_schema["type"] == "string"
    assert "enum" not in field_schema


@pytest.mark.parametrize("model_cls", [OutcomeTimePoint, PrognosticOutcomeTimePoint])
def test_time_point_missing_values(model_cls):
    point = model_cls(time_point_category="Baseline", time_point_detail="At baseline")
    assert point.time_point_value == "NR"
    assert point.time_point_unit == "NR"


def _assert_tabular_outputs(directory, expected_count, detail_column):
    with (directory / "outcomes_test.csv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == expected_count
    for row in rows:
        assert row["time_point_value"] == "6-12"
        assert row["time_point_unit"] == "months"
        assert row[detail_column] == TIME_POINT["time_point_detail"]
    workbook = load_workbook(directory / "study_test.xlsx", read_only=True)
    try:
        values = list(workbook["outcomes"].values)
        assert len(values) == expected_count + 1
        for values_row in values[1:]:
            row = dict(zip(values[0], values_row, strict=True))
            assert row["time_point_value"] == "6-12"
            assert row["time_point_unit"] == "months"
    finally:
        workbook.close()


@pytest.mark.parametrize("study_type", ["RCT", "PrognosticStudy"])
def test_static_time_point_exports(tmp_path, study_type):
    study = _study(study_type)
    export = export_csv if study_type == "RCT" else export_prognostic_csv
    export(study, "example", tmp_path, "test", write_csv=True, write_xlsx=True)
    _assert_tabular_outputs(
        tmp_path / "example",
        3 if study_type == "RCT" else 2,
        "outcome_time_point_detail",
    )
    restored = type(study).model_validate_json(study.model_dump_json())
    assert restored == study
    assert '"time_point_value":"6-12"' in study.model_dump_json()


@pytest.mark.parametrize("study_type", ["RCT", "PrognosticStudy"])
def test_dynamic_time_point_schema_and_exports(tmp_path, study_type):
    csv_path = write_hierarchical_prompts_csv(study_type, tmp_path / "prompts.csv")
    schema = _load_prompt_schema(csv_path)
    unit_field = next(
        field
        for field in schema["OutcomeTimePoint"]
        if field["attribute"] == "time_point_unit"
    )
    assert unit_field["datatype"] == "str"
    unit_field["prompt"] = "Report the time point unit in fortnights."
    dynamic = _build_dynamic_models_from_schema(schema)
    ensure = (
        _ensure_rct_runtime_models
        if study_type == "RCT"
        else _ensure_prognostic_runtime_models
    )
    runtime = ensure(schema, dynamic)
    outcome_name = (
        "Dichotomous_Outcome" if study_type == "RCT" else "HazardRatioOutcome"
    )
    time_point_cls = runtime[outcome_name].model_fields["outcome_time_point"].annotation
    field_schema = time_point_cls.model_json_schema()["properties"]["time_point_unit"]
    assert field_schema["type"] == "string"
    assert "enum" not in field_schema
    assert field_schema["description"] == unit_field["prompt"]
    assert (
        time_point_cls(
            **{**TIME_POINT, "time_point_unit": "fortnights"}
        ).time_point_unit
        == "fortnights"
    )
    study_name = "Study" if study_type == "RCT" else "PrognosticStudy"
    study = runtime[study_name].model_validate(_study(study_type).model_dump())
    pipeline_cls = _build_dynamic_pipeline_for_study_type(study_type, schema)
    prediction = Mock(**study.model_dump())
    if study_type == "RCT":
        prediction.flexible_outcomes = study.model_dump()["other_outcomes"]
    with patch(
        "deet.custom_hierarchical.dspy.Predict",
        return_value=Mock(return_value=prediction),
    ):
        result = pipeline_cls()(context="Synthetic study text")
    assert result.model_dump() == study.model_dump()
    output = tmp_path / "output"
    output.mkdir()
    path = _write_dynamic_study_outputs(
        study,
        schema,
        study_type,
        output,
        "example",
        "test",
        "",
        export_csv_files=True,
        export_xlsx_file=True,
        export_json_file=True,
    )
    _assert_tabular_outputs(
        output / "example", 3 if study_type == "RCT" else 2, "time_point_detail"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "dichotomous_outcomes" if study_type == "RCT" else "hazard_ratio_outcomes"
    assert payload[key][0]["outcome_time_point"] == TIME_POINT


@pytest.mark.parametrize(
    "study_type", ["CochraneRCT", "ObesityRCT", "AnimalRCT", "ClimateCarbonPricing"]
)
def test_other_study_type_prompt_schemas_still_build(tmp_path, study_type):
    csv_path = write_hierarchical_prompts_csv(study_type, tmp_path / "prompts.csv")
    schema = _load_prompt_schema(csv_path)
    pipeline_cls = _build_dynamic_pipeline_for_study_type(study_type, schema)
    assert pipeline_cls() is not None
