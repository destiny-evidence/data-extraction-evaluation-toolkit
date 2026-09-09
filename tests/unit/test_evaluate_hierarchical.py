from unittest.mock import patch

from deet.hierarchical_mvp.evaluate_hierarchical import evaluate_interventions


def test_evaluate_interventions_configures_anthropic_judge_model(tmp_path):
    output_path = tmp_path / "evaluation.csv"

    with (
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.configure_lm"
        ) as configure_lm,
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.load_reference_mapping",
            return_value=[],
        ),
        patch(
            "deet.hierarchical_mvp.evaluate_hierarchical.read_xlsx_sheet_as_dicts",
            return_value=[],
        ),
    ):
        evaluate_interventions(
            "mapping.csv",
            "gold.xlsx",
            output_path,
            llm_model="anthropic/claude-sonnet",
            max_tokens=4096,
        )

    configure_lm.assert_called_once_with("anthropic/claude-sonnet", 4096)
