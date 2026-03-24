"""Tests for tokenisation utilities."""

from unittest.mock import patch

import pytest

from deet.exceptions import LitellmModelNotMappedError
from deet.utils.tokenisation import (
    count_tokens,
    estimate_cost_usd,
    get_model_max_tokens,
    merge_prompt_completion_cost_usd,
    truncate_to_token_limit,
)


def test_get_model_max_tokens_known_model():
    """Test get_model_max_tokens returns int for known model."""
    with patch(
        "deet.utils.tokenisation.litellm.get_max_tokens",
        return_value=4096,
    ):
        result = get_model_max_tokens("gpt-4o-mini")
    assert result is not None
    assert isinstance(result, int)
    assert result == 4096


def test_get_model_max_tokens_unknown_model():
    """Test get_model_max_tokens returns None for unknown model."""
    with patch(
        "deet.utils.tokenisation.litellm.get_max_tokens",
        side_effect=KeyError("unknown"),
    ):
        result = get_model_max_tokens("unknown-model-xyz")
    assert result is None


def test_get_model_max_tokens_unmapped_litellm_message_raises() -> None:
    """Litellm unmapped-model message becomes LitellmModelNotMappedError."""
    msg = (
        "Model test-model isn't mapped yet. Add it here - "
        "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
    )
    with (
        patch(
            "deet.utils.tokenisation.litellm.get_max_tokens",
            side_effect=Exception(msg),
        ),
        pytest.raises(LitellmModelNotMappedError) as exc_info,
    ):
        get_model_max_tokens("test-model")
    assert exc_info.value.args[0] == "test-model"


def test_get_model_max_tokens_other_exception_reraises() -> None:
    """Exceptions that do not match the unmapped-model message propagate."""
    with (
        patch(
            "deet.utils.tokenisation.litellm.get_max_tokens",
            side_effect=RuntimeError("unexpected"),
        ),
        pytest.raises(RuntimeError, match="unexpected"),
    ):
        get_model_max_tokens("any-model")


def test_count_tokens_basic():
    """Test count_tokens returns positive int."""
    with patch(
        "deet.utils.tokenisation.litellm.token_counter",
        return_value=3,
    ):
        result = count_tokens("gpt-4o-mini", "Hello world")
    assert isinstance(result, int)
    assert result == 3


def test_estimate_cost_usd_returns_tuple():
    """Test estimate_cost_usd returns (prompt_cost, completion_cost)."""
    with patch(
        "deet.utils.tokenisation.litellm.cost_per_token",
        return_value=(0.001, 0.002),
    ):
        prompt_cost, completion_cost = estimate_cost_usd(
            "gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
        )
    assert prompt_cost == 0.001
    assert completion_cost == 0.002


def test_estimate_cost_usd_completion_only():
    """Test estimate_cost_usd with completion_tokens only."""
    with patch(
        "deet.utils.tokenisation.litellm.cost_per_token",
        return_value=(None, 0.0001),
    ):
        prompt_cost, completion_cost = estimate_cost_usd(
            "gpt-4o-mini",
            completion_tokens=10,
        )
    assert prompt_cost is None
    assert completion_cost == 0.0001


def test_estimate_cost_usd_on_error_returns_none():
    """Test estimate_cost_usd returns (None, None) when litellm fails."""
    with patch(
        "deet.utils.tokenisation.litellm.cost_per_token",
        side_effect=KeyError("unknown model"),
    ):
        prompt_cost, completion_cost = estimate_cost_usd("unknown-model")
    assert prompt_cost is None
    assert completion_cost is None


def test_estimate_cost_usd_unmapped_registry_exception_returns_none() -> None:
    """Bare Exception with litellm registry wording yields (None, None)."""
    msg = (
        "Model x isn't mapped yet. Add it here - "
        "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
    )
    with patch(
        "deet.utils.tokenisation.litellm.cost_per_token",
        side_effect=Exception(msg),
    ):
        prompt_cost, completion_cost = estimate_cost_usd("x")
    assert prompt_cost is None
    assert completion_cost is None


def test_estimate_cost_usd_other_exception_returns_none() -> None:
    """Non-registry Exception from cost_per_token yields (None, None)."""
    with patch(
        "deet.utils.tokenisation.litellm.cost_per_token",
        side_effect=RuntimeError("unexpected"),
    ):
        prompt_cost, completion_cost = estimate_cost_usd("gpt-4o-mini")
    assert prompt_cost is None
    assert completion_cost is None


@pytest.mark.parametrize(
    ("prompt_c", "completion_c", "expected"),
    [
        (None, None, None),
        (0.01, None, 0.01),
        (None, 0.02, 0.02),
        (0.01, 0.02, 0.03),
    ],
)
def test_merge_prompt_completion_cost_usd(
    prompt_c: float | None,
    completion_c: float | None,
    expected: float | None,
) -> None:
    """merge_prompt_completion_cost_usd sums known parts and handles None."""
    assert merge_prompt_completion_cost_usd(prompt_c, completion_c) == expected


def test_count_tokens_fallback_on_error():
    """Test count_tokens uses char estimate when token_counter fails."""
    with patch(
        "deet.utils.tokenisation.litellm.token_counter",
        side_effect=TypeError("token_counter failed"),
    ):
        result = count_tokens("unknown-model", "Hello world")
    assert result == max(1, 11 // 4)


def test_count_tokens_unmapped_registry_exception_uses_char_estimate() -> None:
    """Bare Exception with litellm registry wording falls back to char estimate."""
    msg = (
        "Model y isn't mapped yet. Add it here - "
        "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
    )
    with patch(
        "deet.utils.tokenisation.litellm.token_counter",
        side_effect=Exception(msg),
    ):
        result = count_tokens("y", "Hello world")
    assert result == max(1, 11 // 4)


def test_truncate_to_token_limit_under_limit():
    """Test truncate_to_token_limit returns unchanged when under limit."""
    text = "Short text"
    with patch(
        "deet.utils.tokenisation.litellm.encode",
        return_value=[1, 2, 3],
    ):
        result = truncate_to_token_limit(text, "gpt-4o-mini", max_tokens=100)
    assert result == text


def test_truncate_to_token_limit_over_limit():
    """Test truncate_to_token_limit truncates when over limit."""
    long_text = " ".join(["word"] * 50)
    truncated = "word word word word word"
    with (
        patch(
            "deet.utils.tokenisation.litellm.encode",
            return_value=list(range(50)),
        ),
        patch(
            "deet.utils.tokenisation.litellm.decode",
            return_value=truncated,
        ),
    ):
        result = truncate_to_token_limit(long_text, "gpt-4o-mini", max_tokens=5)
    assert result == truncated
    assert len(result) < len(long_text)
