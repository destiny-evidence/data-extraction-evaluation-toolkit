import csv
import json
from typing import get_args

import pytest
from pydantic import ValidationError

from deet.custom_hierarchical import (
    _build_dynamic_models_from_schema,
    _build_dynamic_pipeline_for_study_type,
    _load_prompt_schema,
    _serialize_value,
    write_hierarchical_prompts_csv,
)
from deet.hierarchical_mvp.cochrane_postprocessing import postprocess_cochrane_outcomes
from deet.hierarchical_mvp.continuous_statistics import (
    ReportedStatistic,
    revman_statistics,
)
from deet.hierarchical_mvp.rct_registry import RCT_MODEL_MODULES


def test_named_statistic_serialization_preserves_measure_and_context():
    statistic = ReportedStatistic(
        measure="Confidence interval",
        value="8-17",
        basis="Change from baseline",
        details="95% CI for mean change",
    )
    assert _serialize_value([statistic]) == [statistic.model_dump()]
    assert _serialize_value(statistic)["measure"] == "Confidence interval"


@pytest.mark.parametrize("study_type", RCT_MODEL_MODULES)
def test_all_continuous_schemas_require_named_statistic_lists(study_type):
    cls = RCT_MODEL_MODULES[study_type].Continuous_Outcome
    fields = {
        name: field
        for name, field in cls.model_fields.items()
        if name.endswith("_statistics")
    }
    assert len(fields) in (2, 4)
    for field in fields.values():
        assert get_args(field.annotation) == (ReportedStatistic,)
        assert field.is_required()
    assert not {
        "group_mean",
        "group_standard_deviation",
        "mean",
        "standard_deviation",
        "group_a_mean",
        "group_b_mean",
        "mean_a",
        "mean_b",
    }.intersection(cls.model_fields)
    with pytest.raises(ValidationError):
        cls.model_validate({"group_mean": "12", "group_standard_deviation": "2"})


@pytest.mark.parametrize("study_type", RCT_MODEL_MODULES)
def test_prompt_csv_statistic_options_remain_editable(tmp_path, study_type):
    path = write_hierarchical_prompts_csv(study_type, tmp_path / "prompts.csv")
    schema = _load_prompt_schema(path)
    measure = next(
        field
        for field in schema["ReportedStatistic"]
        if field["attribute"] == "measure"
    )
    assert measure["datatype"] == "str"
    measure["prompt"] = "Extract trimmed mean or median absolute deviation."
    models = _build_dynamic_models_from_schema(schema)
    continuous = models["Continuous_Outcome"]
    for name, field in continuous.model_fields.items():
        if not name.endswith("_statistics"):
            continue
        statistic_cls = get_args(field.annotation)[0]
        assert statistic_cls.model_fields["measure"].description == measure["prompt"]
        instance = statistic_cls(measure="Trimmed mean", value="12")
        assert instance.measure == "Trimmed mean"
        assert "enum" not in statistic_cls.model_json_schema()["properties"]["measure"]
        assert field.default_factory is list


@pytest.mark.parametrize("study_type", RCT_MODEL_MODULES)
def test_old_mean_sd_prompt_csvs_are_explicitly_rejected(tmp_path, study_type):
    path = write_hierarchical_prompts_csv(study_type, tmp_path / "prompts.csv")
    schema = _load_prompt_schema(path)
    schema["Continuous_Outcome"] = [
        field
        for field in schema["Continuous_Outcome"]
        if not field["attribute"].endswith("_statistics")
    ]
    schema["Continuous_Outcome"].append(
        {"attribute": "group_mean", "datatype": "str", "prompt": "Mean"}
    )
    with pytest.raises(ValueError, match="Regenerate the prompt CSV"):
        _build_dynamic_pipeline_for_study_type(study_type, schema)


@pytest.mark.parametrize(
    ("measure", "basis", "details", "column", "expected"),
    [
        ("Mean", "Endpoint", "", "Mean", "12"),
        ("Median", "Endpoint", "", "Mean", ""),
        ("Mean change", "Change from baseline", "", "Mean", ""),
        ("Mean", "Change from baseline", "", "Mean", ""),
        ("Mean", "NR", "", "Mean", ""),
        ("Mean", "Endpoint", "Adjusted", "Mean", ""),
        ("Standard deviation", "Endpoint", "", "SD", "12"),
        ("Standard error", "Endpoint", "", "SE", "12"),
        ("Interquartile range", "Endpoint", "", "SD", ""),
        ("Variance", "Endpoint", "", "Variance", "12"),
    ],
)
def test_revman_projection_never_relabels_statistics(
    measure,
    basis,
    details,
    column,
    expected,
):
    statistic = ReportedStatistic(
        measure=measure,
        value="12",
        basis=basis,
        details=details,
    )
    row = revman_statistics([statistic], [statistic])
    assert row[column] == expected
    assert json.loads(row["group_summary_statistics"]) == [statistic.model_dump()]
    assert json.loads(row["group_variability_statistics"]) == [statistic.model_dump()]


def test_revman_does_not_choose_arbitrarily_between_multiple_means():
    statistics = [
        ReportedStatistic(measure="Mean", value="10", basis="Endpoint"),
        ReportedStatistic(measure="Mean", value="12", basis="Endpoint"),
    ]
    row = revman_statistics(statistics, [])
    assert row["Mean"] == ""
    assert '"10"' in row["group_summary_statistics"]
    assert '"12"' in row["group_summary_statistics"]


@pytest.mark.parametrize("study_type", ["CochraneRCT", "CochraneRCT_pairwise"])
def test_cochrane_exports_and_postprocessing_preserve_non_mean_statistics(
    tmp_path,
    study_type,
):
    median = ReportedStatistic(measure="Median", value="12", basis="Endpoint")
    iqr = ReportedStatistic(
        measure="Interquartile range",
        value="8-17",
        basis="Endpoint",
        details="Q1-Q3 for median",
    )
    if study_type.endswith("_pairwise"):
        payload = {
            "outcome_name": "MADRS",
            "arm_a": "Treatment",
            "arm_b": "Control",
            "sample_size_a": "40",
            "sample_size_b": "39",
            "group_a_summary_statistics": [median.model_dump()],
            "group_a_variability_statistics": [iqr.model_dump()],
            "group_b_summary_statistics": [
                ReportedStatistic(
                    measure="Mean", value="16", basis="Endpoint"
                ).model_dump(),
            ],
            "group_b_variability_statistics": [],
        }
    else:
        payload = {
            "outcome_name": "MADRS",
            "group_label": "Treatment",
            "group_N": "40",
            "group_summary_statistics": [median.model_dump()],
            "group_variability_statistics": [iqr.model_dump()],
        }
    model = RCT_MODEL_MODULES[study_type].Continuous_Outcome.model_validate(payload)
    rows = model.to_csv_rows(study="study")
    assert rows[0]["Mean"] == ""
    assert rows[0]["SD"] == ""
    assert json.loads(rows[0]["group_summary_statistics"]) == [median.model_dump()]
    assert json.loads(rows[0]["group_variability_statistics"]) == [iqr.model_dump()]
    if study_type.endswith("_pairwise"):
        assert rows[1]["Mean"] == "16"
        assert rows[1]["group_variability_statistics"] == "[]"
    path = tmp_path / "study.json"
    path.write_text(
        json.dumps(
            {
                "study_characteristics": {"study": "study"},
                "continuous_outcomes": [model.model_dump()],
            }
        ),
        encoding="utf-8",
    )
    output = postprocess_cochrane_outcomes(path)
    with output.open(encoding="utf-8-sig", newline="") as source:
        processed = list(csv.DictReader(source))
    assert len(processed) == len(rows)
    for expected, actual in zip(rows, processed, strict=True):
        assert all(actual[key] == value for key, value in expected.items())


def test_cochrane_postprocessing_keeps_legacy_pairwise_mean_sd_json(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "continuous_outcomes": [
                    {
                        "outcome_name": "MADRS",
                        "arm_a": "Treatment",
                        "arm_b": "Control",
                        "mean_a": "10",
                        "mean_b": "20",
                        "sd_a": "2",
                        "sd_b": "3",
                        "ci_level_a": "0.95",
                        "ci_start_a": "8",
                        "ci_end_a": "12",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = postprocess_cochrane_outcomes(path)
    with output.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["Mean"] for row in rows] == ["10", "20"]
    assert [row["SD"] for row in rows] == ["2", "3"]
    assert rows[0]["CI level"] == "0.95"
    assert rows[0]["CI start"] == "8"
    assert rows[0]["CI end"] == "12"


@pytest.mark.parametrize("study_type", RCT_MODEL_MODULES)
def test_exported_prompt_csv_includes_statistic_list_types(tmp_path, study_type):
    path = write_hierarchical_prompts_csv(study_type, tmp_path / "prompts.csv")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    statistics = [
        row
        for row in rows
        if row["class"] == "Continuous_Outcome"
        and row["attribute"].endswith("_statistics")
    ]
    assert statistics
    assert all(row["datatype"] == "list[ReportedStatistic]" for row in statistics)
    assert {
        row["attribute"] for row in rows if row["class"] == "ReportedStatistic"
    } == {
        "measure",
        "value",
        "basis",
        "details",
    }
