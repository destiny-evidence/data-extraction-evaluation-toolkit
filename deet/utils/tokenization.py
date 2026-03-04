"""Token counting and truncation utilities for LLM context management."""

from __future__ import annotations

import litellm

from deet.logger import logger

# Fallback when model is unknown (e.g. custom Ollama).
_DEFAULT_MAX_TOKENS = 128_000


def get_model_max_tokens(model: str) -> int | None:
    """
    Get the maximum input tokens allowed for a model.

    Uses litellm's model registry. Returns None if the model is unknown.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini", "azure/gpt-4o-mini").

    Returns:
        Maximum input tokens, or None if model not found.

    """
    try:
        result = litellm.get_max_tokens(model)
        return int(result) if result is not None else None
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not get max tokens for model {model}: {e}")
        return None


def count_tokens(model: str, text: str) -> int:
    """
    Count the number of tokens in text for a given model.

    Uses litellm.token_counter with messages format. Falls back to
    a rough character-based estimate (len/4) for unknown models.

    Args:
        model: Model identifier for tokenizer selection.
        text: Text to count tokens for.

    Returns:
        Number of tokens.

    """
    try:
        return litellm.token_counter(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            f"Could not count tokens for model {model}, using char estimate: {e}"
        )
        return max(1, len(text) // 4)


def truncate_to_token_limit(text: str, model: str, max_tokens: int) -> str:
    """
    Truncate text to fit within a token limit.

    Encodes text, truncates the token list, decodes back to string.
    Uses char-based fallback when encode/decode unavailable.

    Args:
        text: Text to truncate.
        model: Model identifier for tokenizer.
        max_tokens: Maximum tokens allowed for the truncated text.

    Returns:
        Truncated text.

    """
    limit = max(1, max_tokens)
    try:
        tokens = litellm.encode(model=model, text=text)
        if len(tokens) <= limit:
            return text
        truncated_tokens = tokens[:limit]
        decoded = litellm.decode(model=model, tokens=truncated_tokens)
        return str(decoded)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not truncate by tokens for model {model}: {e}")
        # Fallback: rough char estimate (~4 chars per token)
        approx_chars = limit * 4
        if len(text) <= approx_chars:
            return text
        return text[:approx_chars]
