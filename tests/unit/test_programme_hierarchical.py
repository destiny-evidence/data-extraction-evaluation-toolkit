import csv
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from deet import custom_hierarchical, main_hierarchical
from deet.hierarchical_mvp.ProgrammeExtraction import ProgrammeExtractionPipeline
from deet.hierarchical_mvp.ProgrammeModel import (
    Intervention,
    Learning,
    Outcome,
    Programme,
    Programme_Characteristics,
)
from deet.hierarchical_mvp.utils import export_programme_csv


def _programme() -> Programme:
    characteristics = Programme_Characteristics(
        countries="Kenya",
        responsible_author="UN Evaluation Office",
        budget="USD 2 million",
    )
    intervention = Intervention(
        name="Skills training",
        description="Training for local partners",
        time_frame="2024-2025",
        resources_used="Facilitators and training materials",
    )
    outcome = Outcome(
        name="Improved delivery capacity",
        description="Partners reported stronger delivery capacity",
        intervention_name=intervention.name,
        unit_of_analysis="Partner organisations",
        sdg_belonging="SDG 17",
        strategic_plan_belonging="Outcome 2",
    )
    learning = Learning(
        outcome_name=outcome.name,
        barrier="Staff turnover",
        enabler="Local ownership",
        lessons_learned="Train multiple staff per partner",
        achievements="Delivery continued after handover",
        recommendation="Expand the training-of-trainers model",
    )
    return Programme(
        programme_characteristics=characteristics,
        interventions=[intervention],
        outcomes=[outcome],
        learnings=[learning],
    )


def test_programme_pipeline_passes_prior_outputs_to_later_steps():
    programme = _programme()
    pipeline = ProgrammeExtractionPipeline()
    pipeline.extract_programme_info = lambda **kwargs: SimpleNamespace(
        programme_characteristics=programme.programme_characteristics,
        interventions=programme.interventions,
    )
    outcomes_calls = []
    learnings_calls = []

    def extract_outcomes(**kwargs):
        outcomes_calls.append(kwargs)
        return SimpleNamespace(outcomes=programme.outcomes)

    def extract_learnings(**kwargs):
        learnings_calls.append(kwargs)
        return SimpleNamespace(learnings=programme.learnings)

    pipeline.extract_outcomes = extract_outcomes
    pipeline.extract_learnings = extract_learnings

    result = pipeline(context="report")

    assert result == programme
    assert outcomes_calls == [
        {
            "context": "report",
            "programme_characteristics": programme.programme_characteristics,
            "interventions": programme.interventions,
        }
    ]
    assert learnings_calls == [
        {
            "context": "report",
            "programme_characteristics": programme.programme_characteristics,
            "interventions": programme.interventions,
            "outcomes": programme.outcomes,
        }
    ]


def test_main_extract_dispatches_programme_pipeline():
    expected = _programme()
    with patch.object(
        main_hierarchical, "ProgrammeExtractionPipeline"
    ) as pipeline_class:
        pipeline_class.return_value.return_value = expected

        result = main_hierarchical.extract("report", "Programme")

    pipeline_class.return_value.assert_called_once_with(context="report")
    assert result is expected


def test_custom_programme_schema_builds_programme_pipeline():
    schema = defaultdict(list)
    for row in custom_hierarchical.build_programme_hierarchical_prompt_rows():
        schema[row["class"]].append(
            {
                "attribute": row["attribute"],
                "prompt": row["prompt"],
                "datatype": row["datatype"],
            }
        )

    pipeline_class = custom_hierarchical._build_dynamic_pipeline_for_study_type(
        "Programme", dict(schema)
    )

    assert pipeline_class.__name__ == "DynamicProgrammeExtractionPipeline"


def test_custom_programme_output_writes_four_tables(tmp_path):
    schema = defaultdict(list)
    for row in custom_hierarchical.build_programme_hierarchical_prompt_rows():
        schema[row["class"]].append(
            {
                "attribute": row["attribute"],
                "prompt": row["prompt"],
                "datatype": row["datatype"],
            }
        )

    custom_hierarchical._write_dynamic_study_outputs(
        _programme(),
        dict(schema),
        "Programme",
        tmp_path,
        "evaluation",
        "20260925_120000",
        "",
    )

    output_dir = tmp_path / "evaluation"
    assert {
        path.name for path in output_dir.glob("*.csv")
    } == {
        "programme_20260925_120000.csv",
        "interventions_20260925_120000.csv",
        "outcomes_20260925_120000.csv",
        "learnings_20260925_120000.csv",
    }


def test_export_programme_csv_writes_four_tables(tmp_path):
    export_programme_csv(
        _programme(),
        "evaluation",
        tmp_path,
        "20260925_120000",
    )

    output_dir = tmp_path / "evaluation"
    expected_headers = {
        "programme": ["countries", "responsible_author", "budget"],
        "interventions": ["name", "description", "time_frame", "resources_used"],
        "outcomes": [
            "name",
            "description",
            "intervention_name",
            "unit_of_analysis",
            "sdg_belonging",
            "strategic_plan_belonging",
        ],
        "learnings": [
            "outcome_name",
            "barrier",
            "enabler",
            "lessons_learned",
            "achievements",
            "recommendation",
        ],
    }
    for table_name, headers in expected_headers.items():
        output_path = output_dir / f"{table_name}_20260925_120000.csv"
        with output_path.open(encoding="utf-8-sig") as csv_file:
            assert next(csv.reader(csv_file)) == headers