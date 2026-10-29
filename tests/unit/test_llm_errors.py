"""Tests for the registry of known litellm errors."""

import httpx
import litellm
import pytest

from deet.exceptions import (
    DocumentSkippedError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextWindowExceededError,
    LLMModelNotFoundError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    RunAbortingError,
    UnsupportedModelParamsError,
)
from deet.extractors.llm_errors import (
    KNOWN_LLM_ERRORS,
    UNSUPPORTED_PARAM_CODES,
    _always,
    translate_llm_error,
)

_COMMON = {"message": "provider said no", "model": "o4-mini", "llm_provider": "openai"}
_FORBIDDEN_RESPONSE = httpx.Response(
    403, request=httpx.Request("POST", "https://example.invalid")
)


@pytest.mark.parametrize(
    ("error", "expected_type", "expected_label"),
    [
        (
            litellm.UnsupportedParamsError(**_COMMON),
            UnsupportedModelParamsError,
            RunAbortingError,
        ),
        (
            litellm.AuthenticationError(**_COMMON),
            LLMAuthenticationError,
            RunAbortingError,
        ),
        (
            litellm.PermissionDeniedError(**_COMMON, response=_FORBIDDEN_RESPONSE),
            LLMAuthenticationError,
            RunAbortingError,
        ),
        (
            litellm.NotFoundError(**_COMMON),
            LLMModelNotFoundError,
            RunAbortingError,
        ),
        (
            litellm.ContentPolicyViolationError(**_COMMON),
            LLMContentFilterError,
            DocumentSkippedError,
        ),
        (
            litellm.ContextWindowExceededError(**_COMMON),
            LLMContextWindowExceededError,
            DocumentSkippedError,
        ),
        (
            litellm.RateLimitError(**_COMMON),
            LLMRateLimitError,
            DocumentSkippedError,
        ),
        (
            litellm.Timeout(**_COMMON),
            LLMTimeoutError,
            DocumentSkippedError,
        ),
    ],
)
def test_translate_llm_error_known_errors(error, expected_type, expected_label):
    """Each known litellm error becomes its deet error with the right label."""
    translated = translate_llm_error(error, model="o4-mini")

    assert type(translated) is expected_type
    assert isinstance(translated, expected_label)
    # litellm's message is shown in full, followed by deet's guidance.
    message = str(translated)
    assert message.startswith(f"{error}\n\n")
    assert len(message) > len(f"{error}\n\n")
    assert "{model}" not in message


@pytest.mark.parametrize("code", sorted(UNSUPPORTED_PARAM_CODES))
def test_translate_llm_error_provider_side_unsupported_param(code):
    """A provider 400 with an unsupported-parameter code stops the run."""
    error = litellm.BadRequestError(
        **_COMMON, body={"code": code, "param": "temperature"}
    )

    translated = translate_llm_error(error, model="my-deployment")

    assert type(translated) is UnsupportedModelParamsError
    assert "my-deployment" in str(translated)


def test_translate_llm_error_quota_exceeded():
    """A rate-limit error caused by an exhausted quota stops the run."""
    # The message litellm 1.102.1 produces for OpenAI's ``insufficient_quota``
    # 429; the error code itself is not kept.
    error = litellm.RateLimitError(
        message=(
            "OpenAIException - You exceeded your current quota, please check "
            "your plan and billing details."
        ),
        model="gpt-4o",
        llm_provider="openai",
    )

    translated = translate_llm_error(error, model="gpt-4o")

    assert type(translated) is LLMQuotaExceededError
    assert isinstance(translated, RunAbortingError)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("LLM call failed"),
        litellm.BadRequestError(**_COMMON),
        litellm.BadRequestError(**_COMMON, body={"code": "something_else"}),
        litellm.APIConnectionError(**_COMMON),
    ],
)
def test_translate_llm_error_unknown_errors(error):
    """Errors that are not in the registry are left for the caller to handle."""
    assert translate_llm_error(error, model="o4-mini") is None


def test_known_llm_errors_have_a_label():
    """Every entry resolves to exactly one of the stop/skip labels."""
    for known in KNOWN_LLM_ERRORS:
        assert issubclass(known.target, RunAbortingError | DocumentSkippedError)


def test_known_llm_errors_are_all_reachable():
    """No entry is shadowed by an earlier unconditional entry for a parent type."""
    for position, known in enumerate(KNOWN_LLM_ERRORS):
        for earlier in KNOWN_LLM_ERRORS[:position]:
            shadowed = earlier.matches is _always and issubclass(
                known.source, earlier.source
            )
            assert not shadowed, f"{known.source.__name__} is unreachable"
