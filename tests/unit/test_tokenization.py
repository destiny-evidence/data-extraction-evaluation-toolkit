"""Tests for tokenization utilities."""

from unittest.mock import patch

from deet.utils.tokenization import (
    count_tokens,
    get_model_max_tokens,
    truncate_to_token_limit,
)


def test_get_model_max_tokens_known_model():
    """Test get_model_max_tokens returns int for known model."""
    result = get_model_max_tokens("gpt-4o-mini")
    assert result is not None
    assert isinstance(result, int)
    assert result > 0


def test_get_model_max_tokens_unknown_model():
    """Test get_model_max_tokens returns None for unknown model."""
    with patch("litellm.get_max_tokens", side_effect=Exception("unknown")):
        result = get_model_max_tokens("unknown-model-xyz")
    assert result is None


def test_count_tokens_basic():
    """Test count_tokens returns positive int."""
    result = count_tokens("gpt-4o-mini", "Hello world")
    assert isinstance(result, int)
    assert result >= 1


def test_count_tokens_fallback_on_error():
    """Test count_tokens uses char estimate when token_counter fails."""
    with patch(
        "deet.utils.tokenization.litellm.token_counter",
        side_effect=Exception("token_counter failed"),
    ):
        result = count_tokens("unknown-model", "Hello world")
    assert result == max(1, 11 // 4)


def test_truncate_to_token_limit_under_limit():
    """Test truncate_to_token_limit returns unchanged when under limit."""
    text = "Short text"
    result = truncate_to_token_limit(text, "gpt-4o-mini", max_tokens=100)
    assert result == text


def test_truncate_to_token_limit_over_limit():
    """Test truncate_to_token_limit truncates when over limit."""
    long_text = " ".join(["word"] * 50)
    result = truncate_to_token_limit(long_text, "gpt-4o-mini", max_tokens=5)
    assert len(result) < len(long_text)
