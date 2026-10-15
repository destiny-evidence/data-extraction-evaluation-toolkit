import json
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel

from deet.hierarchical_mvp import (
    AnimalRCTmodel,
    AnimalRCTmodel_pairwise,
    CochraneRCTmodel,
    CochraneRCTmodel_pairwise,
    ObesityRCTmodel,
    ObesityRCTmodel_pairwise,
    RCTmodel,
    RCTmodel_pairwise,
)


SINGLE_ARM_VARIANTS = [
    (
        "RCT",
        RCTmodel.Study,
        [
            RCTmodel.Dichotomous_Outcome,
            RCTmodel.Continuous_Outcome,
            RCTmodel.Other_Outcome,
        ],
    ),
    (
        "CochraneRCT",
        CochraneRCTmodel.Study,
        [
            CochraneRCTmodel.Dichotomous_Outcome,
            CochraneRCTmodel.Continuous_Outcome,
        ],
    ),
    (
        "ObesityRCT",
        ObesityRCTmodel.Study,
        [
            ObesityRCTmodel.Dichotomous_Outcome,
            ObesityRCTmodel.Continuous_Outcome,
            ObesityRCTmodel.Other_Outcome,
        ],
    ),
    (
        "AnimalRCT",
        AnimalRCTmodel.Study,
        [
            AnimalRCTmodel.Dichotomous_Outcome,
            AnimalRCTmodel.Continuous_Outcome,
            AnimalRCTmodel.Other_Outcome,
        ],
    ),
]

GROUP_VALUES = {
    "Dichotomous_Outcome": {"group_N": "42", "group_Events": "7"},
    "Continuous_Outcome": {
        "group_N": "38",
        "group_summary_statistics": [
            {"measure": "Mean", "value": "4.2", "basis": "Endpoint", "details": ""},
        ],
        "group_variability_statistics": [
            {
                "measure": "Standard deviation",
                "value": "0.8",
                "basis": "Endpoint",
                "details": "",
            },
        ],
    },
    "Other_Outcome": {"group_result": "Improved"},
}

PAIRWISE_VARIANTS = [
    (
        RCTmodel_pairwise,
        [
            RCTmodel_pairwise.Dichotomous_Outcome,
            RCTmodel_pairwise.Continuous_Outcome,
            RCTmodel_pairwise.Other_Outcome,
        ],
    ),
    (
        CochraneRCTmodel_pairwise,
        [
            CochraneRCTmodel_pairwise.Dichotomous_Outcome,
            CochraneRCTmodel_pairwise.Continuous_Outcome,
        ],
    ),
    (
        ObesityRCTmodel_pairwise,
        [
            ObesityRCTmodel_pairwise.Dichotomous_Outcome,
            ObesityRCTmodel_pairwise.Continuous_Outcome,
            ObesityRCTmodel_pairwise.Other_Outcome,
        ],
    ),
    (
        AnimalRCTmodel_pairwise,
        [
            AnimalRCTmodel_pairwise.Dichotomous_Outcome,
            AnimalRCTmodel_pairwise.Continuous_Outcome,
            AnimalRCTmodel_pairwise.Other_Outcome,
        ],
    ),
]


def _sample_payload(model_cls: type[BaseModel]) -> dict[str, object]:
    payload = {}
    for name, field in model_cls.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is list:
            payload[name] = []
        elif get_origin(annotation) is dict:
            payload[name] = {}
        elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
            payload[name] = _sample_payload(annotation)
        elif annotation is bool:
            payload[name] = False
        else:
            payload[name] = f"{name}-value"
    return payload


def _study_outcome_types(study_cls: type[BaseModel]) -> list[type[BaseModel]]:
    return [
        get_args(field.annotation)[0]
        for name, field in study_cls.model_fields.items()
        if name.endswith("_outcomes")
    ]


@pytest.mark.parametrize(
    "variant_name,study_cls,outcome_classes",
    [
        pytest.param(name, study_cls, classes, id=name)
        for name, study_cls, classes in SINGLE_ARM_VARIANTS
    ],
)
def test_study_uses_single_arm_outcome_types(variant_name, study_cls, outcome_classes):
    del variant_name
    assert _study_outcome_types(study_cls) == outcome_classes


@pytest.mark.parametrize(
    "variant_name,study_cls,outcome_cls",
    [
        pytest.param(name, study_cls, outcome_cls, id=f"{name}-{outcome_cls.__name__}")
        for name, study_cls, classes in SINGLE_ARM_VARIANTS
        for outcome_cls in classes
    ],
)
def test_single_arm_outcome_schema_json_and_csv(variant_name, study_cls, outcome_cls):
    del study_cls
    values = GROUP_VALUES[outcome_cls.__name__]
    payload = _sample_payload(outcome_cls)
    payload.update({"outcome_name": "Example outcome", "group_label": "Treatment"})
    payload.update(values)
    outcome = outcome_cls.model_validate(payload)

    properties = outcome_cls.model_json_schema()["properties"]
    assert properties["group_label"]["type"] == "string"
    assert "group_b" not in properties
    assert "group_labels" not in properties
    assert all(not name.startswith("group_b_") for name in properties)

    dumped = json.loads(outcome.model_dump_json())
    assert dumped["group_label"] == "Treatment"
    assert {key: dumped[key] for key in values} == values

    row = outcome.to_csv_row()
    assert list(row) == outcome_cls.csv_fieldnames()
    assert "group_b_label" not in row
    assert all(not name.startswith("group_b_") for name in row)
    assert row["Arm" if variant_name == "CochraneRCT" else "group_label"] == "Treatment"
    if variant_name == "CochraneRCT":
        assert row["Sample size"] == values["group_N"]
        if outcome_cls.__name__ == "Dichotomous_Outcome":
            assert row["Cases"] == values["group_Events"]
        else:
            assert row["Mean"] == "4.2"
            assert row["SD"] == "0.8"
    else:
        assert {
            key: json.loads(row[key]) if isinstance(value, list) else row[key]
            for key, value in values.items()
        } == values


def test_three_madrs_outcomes_remain_distinct_in_study_json_and_csv():
    study = RCTmodel.Study.model_validate(_sample_payload(RCTmodel.Study))
    madrs_outcomes = [
        RCTmodel.Continuous_Outcome.model_validate(
            {
                **_sample_payload(RCTmodel.Continuous_Outcome),
                "outcome_name": name,
                "group_label": "Intervention",
                "group_N": str(40 - index),
                "group_summary_statistics": [
                    {"measure": "Mean", "value": str(20 - index)},
                ],
                "group_variability_statistics": [
                    {"measure": "Standard deviation", "value": "5"},
                ],
            }
        )
        for index, name in enumerate(
            [
                "MADRS total score",
                "MADRS response",
                "MADRS remission",
            ]
        )
    ]
    study.continuous_outcomes.extend(madrs_outcomes)

    serialized = json.loads(study.model_dump_json())
    restored = RCTmodel.Study.model_validate(serialized)
    assert [outcome.outcome_name for outcome in restored.continuous_outcomes] == [
        "MADRS total score",
        "MADRS response",
        "MADRS remission",
    ]
    assert [
        outcome.to_csv_row()["group_N"] for outcome in restored.continuous_outcomes
    ] == [
        "40",
        "39",
        "38",
    ]


@pytest.mark.parametrize(
    "module,outcome_classes",
    [
        pytest.param(module, classes, id=module.__name__.rsplit(".", 1)[-1])
        for module, classes in PAIRWISE_VARIANTS
    ],
)
def test_pairwise_modules_preserve_arm_structure_and_exports(module, outcome_classes):
    for outcome_cls in outcome_classes:
        payload = _sample_payload(outcome_cls)
        if "group_labels" in outcome_cls.model_fields:
            payload["group_labels"] = {
                "group_a": "Treatment",
                "group_b": "Control",
            }
        outcome = outcome_cls.model_validate(payload)

        if module is CochraneRCTmodel_pairwise:
            rows = outcome.to_csv_rows(study="study-id")
            assert len(rows) == 2
            assert [row["Arm"] for row in rows] == [outcome.arm_a, outcome.arm_b]
        elif module is AnimalRCTmodel_pairwise:
            row = outcome.to_csv_row()
            assert "intervention_to_induce_name" in row
            assert "intervention_to_assess_name" in row
        else:
            row = outcome.to_csv_row()
            assert row["group_a_label"] == "Treatment"
            assert row["group_b_label"] == "Control"
            assert any(name.startswith("group_a_") for name in row)
            assert any(name.startswith("group_b_") for name in row)
