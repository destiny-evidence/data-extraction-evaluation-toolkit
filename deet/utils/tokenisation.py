"""Token counting and truncation utilities for LLM context management."""

import re

import litellm
from litellm.exceptions import NotFoundError as LitellmNotFoundError
from loguru import logger

from deet.exceptions import LitellmModelNotMappedError

# litellm raises NotFoundError for unknown models and KeyError / ValueError
# from internal dict lookups.  We catch only these so genuinely unexpected
# errors still propagate.
_LITELLM_LOOKUP_ERRORS = (LitellmNotFoundError, KeyError, ValueError)

# Tokeniser operations can additionally raise TypeError (wrong input type)
# or AttributeError (missing tokeniser method).
_LITELLM_TOKENISER_ERRORS = (*_LITELLM_LOOKUP_ERRORS, TypeError, AttributeError)

# litellm.get_max_tokens raises bare Exception with this wording for unmapped models.
_LITELLM_UNMAPPED_MODEL_MSG = re.compile(
    r"isn['\u2019]?t mapped yet|model_prices_and_context_window\.json",
    re.IGNORECASE,
)


def get_model_max_tokens(model: str) -> int | None:
    """
    Get the maximum input tokens allowed for a model.

    Uses litellm's model registry. Returns None if the model is unknown.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini", "azure/gpt-4o-mini").

    Returns:
        Maximum input tokens, or None if model not found.

    Raises:
        LitellmModelNotMappedError: When litellm raises a bare ``Exception``
            whose message indicates an unmapped model (registry message).
        Exception: Any other error from litellm is re-raised.

    """
    try:
        result = litellm.get_max_tokens(model)
        return int(result) if result is not None else None
    except _LITELLM_LOOKUP_ERRORS as e:
        logger.debug(f"Could not get max tokens for model {model}: {e}")
        return None
    except Exception as e:
        if _LITELLM_UNMAPPED_MODEL_MSG.search(str(e)):
            logger.debug(f"Litellm model not in registry for max tokens: {model}: {e}")
            raise LitellmModelNotMappedError(model) from e
        raise


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
    except _LITELLM_TOKENISER_ERRORS as e:
        logger.warning(
            f"Could not count tokens for model {model}, using char estimate: {e}"
        )
    except Exception as e:  # noqa: BLE001
        # litellm token_counter may raise bare Exception (registry / provider).
        if _LITELLM_UNMAPPED_MODEL_MSG.search(str(e)):
            logger.debug(
                f"Could not count tokens for model {model} (unmapped in registry): {e}"
            )
        else:
            logger.warning(
                f"Could not count tokens for model {model}, using char estimate: {e}"
            )
    return max(1, len(text) // 4)


def estimate_cost_usd(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> tuple[float | None, float | None]:
    """
    Estimate cost in USD for prompt and completion tokens.

    Uses litellm's cost_per_token. Returns (prompt_cost_usd, completion_cost_usd).
    Either or both may be None if the model is unknown or the call fails.
    Any other ``Exception`` from litellm (e.g. provider HTTP errors) is logged
    at debug and also yields ``(None, None)`` so cost estimation stays
    best-effort for callers.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini", "azure/gpt-4o-mini").
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.

    Returns:
        Tuple of (prompt_cost_usd, completion_cost_usd). Either can be None.

    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return (
            float(prompt_cost) if prompt_cost is not None else None,
            float(completion_cost) if completion_cost is not None else None,
        )
    except _LITELLM_LOOKUP_ERRORS as e:
        logger.debug(f"Could not estimate cost for model {model}: {e}")
        return (None, None)
    except Exception as e:  # noqa: BLE001
        # litellm cost_per_token may raise bare Exception (registry / provider).
        if _LITELLM_UNMAPPED_MODEL_MSG.search(str(e)):
            logger.debug(
                f"Could not estimate cost for model {model} (unmapped in registry): {e}"
            )
        else:
            logger.debug(f"Could not estimate cost for model {model}: {e}")
        return (None, None)


def merge_prompt_completion_cost_usd(
    prompt_cost_usd: float | None,
    completion_cost_usd: float | None,
) -> float | None:
    """
    Combine litellm prompt and completion cost parts into a single total.

    Either part may be None when the model table lacks that component or
    estimation failed for that side.

    Args:
        prompt_cost_usd: Estimated USD cost for prompt tokens, or None.
        completion_cost_usd: Estimated USD cost for completion tokens, or None.

    Returns:
        Sum when at least one part is known, otherwise None.

    """
    if prompt_cost_usd is not None and completion_cost_usd is not None:
        return prompt_cost_usd + completion_cost_usd
    if prompt_cost_usd is not None:
        return prompt_cost_usd
    if completion_cost_usd is not None:
        return completion_cost_usd
    return None


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
    except _LITELLM_TOKENISER_ERRORS as e:
        logger.debug(f"Could not truncate by tokens for model {model}: {e}")
    # Fallback: rough char estimate (~4 chars per token)
    approx_chars = limit * 4
    if len(text) <= approx_chars:
        return text
    return text[:approx_chars]
