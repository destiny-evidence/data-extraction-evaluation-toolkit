"""Unit tests for extraction evaluation metrics."""

from functools import partial

import numpy as np
import pytest
from sklearn.metrics import (  # type:ignore[import-untyped]
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from deet.data_models.base import AttributeType
from deet.evaluators.embeddings import DEFAULT_EMBEDDING_MODEL_NAME
from deet.evaluators.metrics import (
    EvaluationMetricSettings,
    edit_distance_match_rate,
    get_metrics_for_attribute_type,
    semantic_match_rate,
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
    """Empty aligned lists raise, matching sklearn empty-input behaviour."""
    with pytest.raises(ValueError, match="non-empty"):
        edit_distance_match_rate([], [])


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
    assert "semantic_match_rate" not in string_metrics

    for attr_type in (AttributeType.INTEGER, AttributeType.FLOAT):
        numeric_metrics = get_metrics_for_attribute_type(attr_type)
        assert "accuracy" in numeric_metrics
        assert "mean_absolute_error" in numeric_metrics
        assert "mean_absolute_percentage_error" in numeric_metrics
        assert "semantic_match_rate" not in numeric_metrics
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
        "provider: azure\nmodel: gpt-5.6-luna\nmax_context_tokens: 1000\n",
        encoding="utf-8",
    )
    default_config = DataExtractionConfig.from_yaml(default_path)
    assert default_config.edit_distance_match_threshold == 0.90

    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        "provider: azure\nmodel: gpt-5.6-luna\nmax_context_tokens: 1000\n"
        "edit_distance_match_threshold: 0.85\n",
        encoding="utf-8",
    )
    custom_config = DataExtractionConfig.from_yaml(custom_path)
    assert custom_config.edit_distance_match_threshold == 0.85


def test_semantic_match_rate_normalises_before_embedding(monkeypatch) -> None:
    """Whitespace and case are folded before strings are embedded."""
    seen: dict[str, object] = {}

    def _encode(texts: list[str], *, model_name: str = "") -> np.ndarray:
        seen["texts"] = list(texts)
        seen["model"] = model_name
        return np.eye(len(texts), dtype=np.float64)

    monkeypatch.setattr("deet.evaluators.embeddings.encode_texts", _encode)
    rate = semantic_match_rate(
        ["Odds  Ratio", "hypertension"],
        ["odds ratio", "hypertention"],
        model_name="unit-test-model",
    )
    assert seen["texts"] == ["odds ratio", "hypertension", "hypertention"]
    assert seen["model"] == "unit-test-model"
    assert rate == 0.5


def test_semantic_match_rate_respects_threshold(monkeypatch) -> None:
    """Pairs below the cosine threshold do not count as matches."""

    def _encode(texts: list[str], *, model_name: str = "") -> np.ndarray:
        table = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.6, 0.8]),
        }
        return np.vstack([table[text] for text in texts])

    monkeypatch.setattr("deet.evaluators.embeddings.encode_texts", _encode)
    assert semantic_match_rate(["a"], ["b"]) == 0.0
    assert semantic_match_rate(["a"], ["b"], threshold=0.5) == 1.0


def test_semantic_match_rate_rejects_none_predictions() -> None:
    """None predictions raise, consistent with edit-distance behaviour."""
    with pytest.raises(TypeError, match="None predictions"):
        semantic_match_rate(["a", "b"], ["a", None])


def test_semantic_match_rate_rejects_length_mismatch() -> None:
    """Mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        semantic_match_rate(["a", "b"], ["a"])


def test_semantic_match_rate_empty_lists() -> None:
    """Empty aligned lists raise, matching sklearn empty-input behaviour."""
    with pytest.raises(ValueError, match="non-empty"):
        semantic_match_rate([], [])


def test_get_metrics_includes_semantic_match_only_when_enabled() -> None:
    """STRING metrics gain semantic_match_rate when the flag is on."""
    enabled = get_metrics_for_attribute_type(
        AttributeType.STRING,
        settings=EvaluationMetricSettings(
            enable_semantic_metrics=True,
            semantic_match_threshold=0.5,
            embedding_model_name="unit-test-model",
        ),
    )
    assert "semantic_match_rate" in enabled
    semantic_metric = enabled["semantic_match_rate"]
    assert isinstance(semantic_metric, partial)
    assert semantic_metric.keywords["threshold"] == 0.5
    assert semantic_metric.keywords["model_name"] == "unit-test-model"


def test_data_extraction_config_semantic_settings_from_yaml(tmp_path) -> None:
    """Semantic settings default off and can be set from the extraction YAML."""
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "provider: azure\nmodel: gpt-5.6-luna\nmax_context_tokens: 1000\n",
        encoding="utf-8",
    )
    default_config = DataExtractionConfig.from_yaml(default_path)
    assert default_config.enable_semantic_metrics is False
    assert default_config.semantic_match_threshold == 0.85
    assert default_config.embedding_model_name == DEFAULT_EMBEDDING_MODEL_NAME

    custom_path = tmp_path / "custom.yaml"
    custom_path.write_text(
        "provider: azure\nmodel: gpt-5.6-luna\nmax_context_tokens: 1000\n"
        "enable_semantic_metrics: true\n"
        "semantic_match_threshold: 0.80\n"
        "embedding_model_name: BAAI/bge-small-en-v1.5\n",
        encoding="utf-8",
    )
    custom_config = DataExtractionConfig.from_yaml(custom_path)
    assert custom_config.enable_semantic_metrics is True
    assert custom_config.semantic_match_threshold == 0.80
    assert custom_config.embedding_model_name == "BAAI/bge-small-en-v1.5"
