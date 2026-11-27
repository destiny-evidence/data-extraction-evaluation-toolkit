"""Registry of known litellm errors and the deet errors they are reported as."""

from collections.abc import Callable, Collection
from dataclasses import dataclass

import litellm

from deet.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextWindowExceededError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    UnsupportedModelParamsError,
    UserFacingError,
)


def _has_error_code(codes: Collection[str]) -> Callable[[Exception], bool]:
    """Build a condition matching the provider's error ``code`` (OpenAI/Azure)."""

    def matches(error: Exception) -> bool:
        return getattr(error, "code", None) in codes

    return matches


def _has_body_error_code(codes: Collection[str]) -> Callable[[Exception], bool]:
    """
    Build a condition matching the ``code`` in the provider's error body.

    For errors where litellm overwrites ``code`` (``RateLimitError`` always sets
    it to "429"). litellm keeps the provider's body from 1.103.0, the floor in
    ``pyproject.toml``.
    """

    def matches(error: Exception) -> bool:
        body = getattr(error, "body", None)
        return isinstance(body, dict) and body.get("code") in codes

    return matches


def _always(_error: Exception) -> bool:
    return True


UNSUPPORTED_PARAM_CODES = frozenset({"unsupported_value", "unsupported_parameter"})
QUOTA_EXCEEDED_CODES = frozenset({"insufficient_quota"})


@dataclass(frozen=True)
class KnownLLMError:
    """
    How one litellm error is reported by deet.

    Args:
        source: The litellm exception type to match.
        target: The deet error raised instead. Its parent class decides whether
            the run stops (``RunAbortingError``) or the document is skipped
            (``DocumentSkippedError``).
        guidance: What the user should do, appended to litellm's own message.
            May reference ``{model}``.
        matches: Extra condition on the error, for when the type is too broad.

    """

    source: type[Exception]
    target: type[UserFacingError]
    guidance: str
    matches: Callable[[Exception], bool] = _always


_UNSUPPORTED_PARAMS_GUIDANCE = (
    "Model '{model}' does not support one of the configured parameters (see "
    "the error above). Change that value in your extraction config and "
    "re-run. (litellm's `drop_params` option is not exposed by deet.)"
)

# First match wins. Most litellm errors here subclass ``BadRequestError``, so
# entries for a subclass must come before any entry for its parent.
KNOWN_LLM_ERRORS: tuple[KnownLLMError, ...] = (
    KnownLLMError(
        source=litellm.UnsupportedParamsError,
        target=UnsupportedModelParamsError,
        guidance=_UNSUPPORTED_PARAMS_GUIDANCE,
    ),
    KnownLLMError(
        source=litellm.ContentPolicyViolationError,
        target=LLMContentFilterError,
        guidance="The provider's content filter rejected this document.",
    ),
    KnownLLMError(
        source=litellm.ContextWindowExceededError,
        target=LLMContextWindowExceededError,
        guidance=(
            "This document does not fit in the context window of '{model}'. "
            "Lower `max_context_tokens` and set `truncate_on_overflow: true` "
            "in your extraction config, or use a model with a larger context "
            "window."
        ),
    ),
    KnownLLMError(
        source=litellm.BadRequestError,
        target=UnsupportedModelParamsError,
        guidance=_UNSUPPORTED_PARAMS_GUIDANCE,
        matches=_has_error_code(UNSUPPORTED_PARAM_CODES),
    ),
    KnownLLMError(
        source=litellm.AuthenticationError,
        target=LLMAuthenticationError,
        guidance=(
            "The provider rejected your credentials. Check the API key and "
            "endpoint settings for '{model}' in your environment and re-run."
        ),
    ),
    KnownLLMError(
        source=litellm.PermissionDeniedError,
        target=LLMAuthenticationError,
        guidance=(
            "Your credentials do not have access to '{model}'. Check the "
            "permissions of your API key or pick another model, then re-run."
        ),
    ),
    KnownLLMError(
        source=litellm.NotFoundError,
        target=LLMModelNotFoundError,
        guidance=(
            "The provider could not find '{model}'. Check the model (or "
            "deployment) name and endpoint in your extraction config and "
            "re-run."
        ),
    ),
    KnownLLMError(
        source=litellm.RateLimitError,
        target=LLMQuotaExceededError,
        guidance=(
            "The provider account for '{model}' has run out of quota. Check "
            "your plan and billing details, then re-run."
        ),
        matches=_has_body_error_code(QUOTA_EXCEEDED_CODES),
    ),
    KnownLLMError(
        source=litellm.RateLimitError,
        target=LLMRateLimitError,
        guidance=(
            "The provider rate-limited this request and retries ran out. "
            "Lower `max_workers` or raise `max_retries` in your extraction "
            "config if this happens for many documents."
        ),
    ),
    KnownLLMError(
        source=litellm.Timeout,
        target=LLMTimeoutError,
        guidance=(
            "The request to '{model}' timed out and retries ran out. Raise "
            "`max_retries` in your extraction config if this happens for many "
            "documents."
        ),
    ),
)


def translate_llm_error(error: Exception, *, model: str) -> UserFacingError | None:
    """
    Return the deet error for a known litellm error, or None if it is unknown.

    The message is litellm's own text (verbatim, so it survives wording
    changes) followed by deet's guidance.
    """
    for known in KNOWN_LLM_ERRORS:
        if isinstance(error, known.source) and known.matches(error):
            guidance = known.guidance.format(model=model)
            return known.target(f"{error}\n\n{guidance}")
    return None
