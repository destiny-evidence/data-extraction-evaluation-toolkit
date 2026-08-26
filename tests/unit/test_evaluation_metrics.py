"""Unit tests for extraction evaluation metrics."""

import pytest
from sklearn.metrics import (  # type:ignore[import-untyped]
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from deet.data_models.base import AttributeType
from deet.evaluators.metrics import (
    EvaluationMetricSettings,
    edit_distance_match_rate,
    get_metrics_for_attribute_type,
)
from deet.extractors.llm_data_extractor import DataExtractionConfig


def test_edit_distance_match_rate_near_match_above_threshold() -> None:
    """A single-character typo near-match scores as a match at default threshold."""
    # normalised similarity of hypertension/hypertention is ~0.9167 >= 0.90
    rate = edit_distance_match_rate(
        ["hypertension", "odds ratio"],
        ["hypertention", "odds ratio"],
    )
    assert rate == 1.0


def test_edit_distance_match_rate_below_threshold() -> None:
    """Low-similarity pairs do not count as matches."""
    rate = edit_distance_match_rate(
        ["abc"],
        ["xyz"],
        threshold=0.90,
    )
    assert rate == 0.0


def test_edit_distance_match_rate_uses_normalisation() -> None:
    """Whitespace / case differences do not prevent a match after normalisation."""
    rate = edit_distance_match_rate(
        ["Odds  Ratio"],
        ["odds ratio"],
    )
    assert rate == 1.0


def test_edit_distance_match_rate_rejects_none_predictions() -> None:
    """None predictions raise, consistent with binary-metric failure behaviour."""
    with pytest.raises(TypeError, match="None predictions"):
        edit_distance_match_rate(["a", "b"], ["a", None])


def test_edit_distance_match_rate_rejects_length_mismatch() -> None:
    """Mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        edit_distance_match_rate(["a", "b"], ["a"])


def test_edit_distance_match_rate_empty_lists() -> None:
    """Empty aligned lists yield 0.0."""
    assert edit_distance_match_rate([], []) == 0.0


def test_mean_absolute_error_rounding_vs_hallucination() -> None:
    """Small rounding error yields small MAE; hallucination yields large MAE."""
    rounding_mae = mean_absolute_error([100.0, 50.0], [100.5, 49.5])
    hallucination_mae = mean_absolute_error([100.0, 50.0], [0.05, 2024.0])
    assert rounding_mae == pytest.approx(0.5)
    assert hallucination_mae > rounding_mae
    assert hallucination_mae == pytest.approx((99.95 + 1974.0) / 2)


def test_mean_absolute_percentage_error_rounding_vs_hallucination() -> None:
    """MAPE distinguishes relative scale of errors."""
    rounding_mape = mean_absolute_percentage_error([100.0], [101.0])
    hallucination_mape = mean_absolute_percentage_error([100.0], [1.0])
    assert rounding_mape == pytest.approx(0.01)
    assert hallucination_mape == pytest.approx(0.99)


def test_numeric_metrics_reject_none_predictions() -> None:
    """None predictions are not accepted by sklearn MAE/MAPE."""
    with pytest.raises((TypeError, ValueError)):
        mean_absolute_error([10.0, 20.0], [12.0, None])
    with pytest.raises((TypeError, ValueError)):
        mean_absolute_percentage_error([10.0, 20.0], [12.0, None])


def test_get_metrics_for_attribute_type_registers_new_metrics() -> None:
    """STRING / INTEGER / FLOAT registries expose the new extraction metrics."""
    string_metrics = get_metrics_for_attribute_type(AttributeType.STRING)
    assert "accuracy" in string_metrics
    assert "edit_distance_match_rate" in string_metrics

    for attr_type in (AttributeType.INTEGER, AttributeType.FLOAT):
        numeric_metrics = get_metrics_for_attribute_type(attr_type)
        assert "accuracy" in numeric_metrics
        assert "mean_absolute_error" in numeric_metrics
        assert "mean_absolute_percentage_error" in numeric_metrics
        assert numeric_metrics["mean_absolute_error"] is mean_absolute_error
        assert (
            numeric_metrics["mean_absolute_percentage_error"]
            is mean_absolute_percentage_error
        )


def test_get_metrics_for_attribute_type_respects_threshold_settings() -> None:
    """Custom edit-distance threshold is applied via settings."""
    strict = get_metrics_for_attribute_type(
        AttributeType.STRING,
        settings=EvaluationMetricSettings(edit_distance_match_threshold=0.99),
    )
    # hypertension vs hypertention ~0.9167: matches at 0.90, not at 0.99
    assert strict["edit_distance_match_rate"](["hypertension"], ["hypertention"]) == 0.0


def test_data_extraction_config_edit_distance_threshold_from_yaml(tmp_path) -> None:
    """Config YAML omits threshold → 0.90; explicit value is honoured."""
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "provider: azure\nmodel: gpt-4o-mini\nmax_context_tokens: 1000\n",
        encoding="utf-8",
    )
    default_config = DataExtractionConfig.from_yaml(default_path)
    assert default_config.edit_distance_match_threshold == 0.90

    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        "provider: azure\nmodel: gpt-4o-mini\nmax_context_tokens: 1000\n"
        "edit_distance_match_threshold: 0.85\n",
        encoding="utf-8",
    )
    custom_config = DataExtractionConfig.from_yaml(custom_path)
    assert custom_config.edit_distance_match_threshold == 0.85
