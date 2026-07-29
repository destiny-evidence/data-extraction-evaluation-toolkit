"""Unit tests for extraction evaluation metrics."""

import pytest
from loguru import logger

from deet.data_models.base import AttributeType
from deet.data_models.evaluation import (
    EvaluationMetricSettings,
    edit_distance_match_rate,
    filter_valid_pairs,
    get_metrics_for_attribute_type,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from deet.extractors.llm_data_extractor import DataExtractionConfig


def test_filter_valid_pairs_drops_none_predictions() -> None:
    """None predictions are removed while keeping gold/pred alignment."""
    y_true = ["a", "b", "c", "d"]
    y_pred = ["a", None, "c", None]
    filtered_true, filtered_pred = filter_valid_pairs(y_true, y_pred)
    assert filtered_true == ["a", "c"]
    assert filtered_pred == ["a", "c"]


def test_filter_valid_pairs_rejects_length_mismatch() -> None:
    """Mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        filter_valid_pairs([1, 2], [1])


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


def test_edit_distance_match_rate_filters_none_and_empty() -> None:
    """None predictions are dropped; all-None yields 0.0."""
    assert edit_distance_match_rate(["a", "b"], ["a", None]) == 1.0
    assert edit_distance_match_rate(["a"], [None]) == 0.0


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


def test_mean_absolute_percentage_error_skips_zero_gold() -> None:
    """Zero-gold pairs are skipped with a warning; remaining pairs are scored."""
    messages: list[str] = []
    logger_id = logger.add(messages.append, level="WARNING")
    try:
        mape = mean_absolute_percentage_error([0.0, 100.0], [5.0, 110.0])
    finally:
        logger.remove(logger_id)

    assert mape == pytest.approx(0.10)
    assert any("skipped 1 pair(s) with zero gold value" in m for m in messages)


def test_mean_absolute_percentage_error_all_zero_gold_raises() -> None:
    """MAPE raises when every gold value is zero after filtering."""
    with pytest.raises(ValueError, match="No valid pairs for MAPE"):
        mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0])


def test_numeric_metrics_filter_none_predictions() -> None:
    """None predictions are excluded from MAE/MAPE denominators."""
    assert mean_absolute_error([10.0, 20.0], [12.0, None]) == pytest.approx(2.0)
    assert mean_absolute_percentage_error([10.0, 20.0], [12.0, None]) == pytest.approx(
        0.2
    )


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
