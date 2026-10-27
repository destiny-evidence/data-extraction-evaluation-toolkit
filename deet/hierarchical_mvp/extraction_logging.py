"""Collect per-call DSPy token usage without changing extraction schemas."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Concatenate, Literal
from uuid import uuid4

import dspy
from dspy.utils.callback import BaseCallback
from pydantic import BaseModel, Field

from deet.logger import logger

if TYPE_CHECKING:
    from pathlib import Path


class TokenUsage(BaseModel):
    """Normalized counts; unknown counts remain null, never silently zero."""

    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    cached_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    non_cached_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    cache_write_input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return value
    return {}


def normalize_usage(raw: Mapping[str, Any]) -> TokenUsage:
    """Normalize OpenAI, Responses, and native Anthropic usage payloads."""
    details = _mapping(
        raw.get("prompt_tokens_details", raw.get("input_tokens_details"))
    )
    output_details = _mapping(
        raw.get("completion_tokens_details", raw.get("output_tokens_details"))
    )
    cached = details.get("cached_tokens", raw.get("cache_read_input_tokens"))
    writes = details.get(
        "cache_write_tokens",
        details.get("cache_creation_tokens", raw.get("cache_creation_input_tokens")),
    )
    usage = TokenUsage(
        input_tokens=raw.get("prompt_tokens", raw.get("input_tokens")),
        cached_input_tokens=cached,
        cache_write_input_tokens=writes,
        output_tokens=raw.get("completion_tokens", raw.get("output_tokens")),
        reasoning_tokens=output_details.get("reasoning_tokens"),
    )
    # Native Anthropic input excludes cache reads/writes; LiteLLM prompt_tokens
    # already includes them, as does OpenAI/Responses input_tokens.
    if (
        "prompt_tokens" not in raw
        and "cache_read_input_tokens" in raw
        and usage.input_tokens is not None
    ):
        if usage.cached_input_tokens is None or usage.cache_write_input_tokens is None:
            usage.input_tokens = None
        else:
            usage.input_tokens += (
                usage.cached_input_tokens + usage.cache_write_input_tokens
            )
    if usage.input_tokens is not None and usage.cached_input_tokens is not None:
        remaining = usage.input_tokens - usage.cached_input_tokens
        if remaining < 0 or (
            usage.cache_write_input_tokens is not None
            and usage.cache_write_input_tokens > remaining
        ):
            message = "Cache token counts exceed total input tokens."
            raise ValueError(message)
        usage.non_cached_input_tokens = remaining
    return usage


class UsageCall(BaseModel):
    """One LM invocation, including local-cache hits and failed invocations."""

    call_id: str
    stage: str
    attempt: int
    model: str
    response_model: str | None = None
    timestamp: str
    status: Literal["running", "completed", "failed"] = "running"
    local_cache_hit: bool | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None


class ExtractionUsageLog(BaseModel):
    """Usage for a single study extraction; contains no document or prompt text."""

    schema_version: int = 1
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    study_name: str
    study_type: str
    model: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    error_type: str | None = None
    calls: list[UsageCall] = Field(default_factory=list)
    totals: TokenUsage = Field(default_factory=TokenUsage)
    usage_complete: bool = False
    logging_errors: list[str] = Field(default_factory=list)

    def summarize(self) -> None:
        """Sum each count only when all invocations report that count."""
        values: dict[str, int | None] = {}
        for field in TokenUsage.model_fields:
            counts = [getattr(call.usage, field) for call in self.calls]
            values[field] = (
                None if any(count is None for count in counts) else sum(counts)
            )
        self.totals = TokenUsage(**values)
        self.usage_complete = not self.logging_errors and all(
            getattr(self.totals, field) is not None
            for field in (
                "input_tokens",
                "cached_input_tokens",
                "non_cached_input_tokens",
                "output_tokens",
            )
        )


def _record_logging_error(
    log: ExtractionUsageLog | None, operation: str, exception: Exception
) -> None:
    if log is not None:
        log.logging_errors.append(f"{operation}: {type(exception).__name__}")
        log.usage_complete = False
    logger.exception(
        f"Token logging failed during {operation}; prediction saving will continue."
    )


def _guard_callback[**P](
    method: Callable[Concatenate[UsageCallback, P], None],
) -> Callable[Concatenate[UsageCallback, P], None]:
    @wraps(method)
    def guarded(self: UsageCallback, *args: P.args, **kwargs: P.kwargs) -> None:
        try:
            method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            _record_logging_error(self.log, method.__name__, exc)

    return guarded


class UsageCallback(BaseCallback):
    """Attribute LM history entries to their enclosing predictor signatures."""

    def __init__(self, log: ExtractionUsageLog) -> None:
        """Create an isolated collector for one extraction run."""
        self.log = log
        self._stages: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
            f"extraction_stages_{log.run_id}", default=()
        )
        self._pending: dict[str, tuple[dspy.BaseLM, object, UsageCall]] = {}

    @_guard_callback
    def on_module_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        """Track predictors, including dynamic and nested signatures."""
        if isinstance(instance, dspy.Predict):
            self._stages.set(
                (*self._stages.get(), (call_id, instance.signature.__name__))
            )

    @_guard_callback
    def on_module_end(
        self, call_id: str, outputs: object, exception: Exception | None = None
    ) -> None:
        """Remove the completed predictor from the current stage stack."""
        self._stages.set(
            tuple(item for item in self._stages.get() if item[0] != call_id)
        )

    @_guard_callback
    def on_lm_start(
        self, call_id: str, instance: object, inputs: dict[str, Any]
    ) -> None:
        """Create a record before an invocation so failures are retained."""
        if not isinstance(instance, dspy.BaseLM):
            message = "Usage logging requires a DSPy BaseLM instance."
            raise TypeError(message)
        stages = self._stages.get()
        stage = stages[-1][1] if stages else "unattributed"
        call = UsageCall(
            call_id=call_id,
            stage=stage,
            attempt=1 + sum(item.stage == stage for item in self.log.calls),
            model=instance.model,
            timestamp=datetime.now(UTC).isoformat(),
        )
        previous = instance.history[-1].get("uuid") if instance.history else None
        self._pending[call_id] = (instance, previous, call)
        self.log.calls.append(call)

    @_guard_callback
    def on_lm_end(
        self, call_id: str, outputs: object, exception: Exception | None = None
    ) -> None:
        """Read usage, distinguishing DSPy response-cache hits from API reads."""
        instance, previous, call = self._pending.pop(call_id)
        call.status = "failed" if exception else "completed"
        call.error_type = type(exception).__name__ if exception else None
        entry = instance.history[-1] if instance.history else {}
        if not entry or entry.get("uuid") == previous:
            logger.warning(f"No response usage available for {call.stage}.")
            return
        call.response_model = entry.get("response_model")
        call.local_cache_hit = bool(getattr(entry.get("response"), "cache_hit", False))
        call.raw_usage = dict(_mapping(entry.get("usage")))
        call.usage = (
            TokenUsage(**dict.fromkeys(TokenUsage.model_fields, 0))
            if call.local_cache_hit
            else normalize_usage(call.raw_usage)
        )
        if call.usage.cached_input_tokens is None:
            logger.warning(f"Cached token usage is unavailable for {call.stage}.")


def _finalize_log(log: ExtractionUsageLog, path: Path | None) -> None:
    try:
        log.finished_at = datetime.now(UTC).isoformat()
        log.summarize()
    except Exception as exc:  # noqa: BLE001
        _record_logging_error(log, "aggregation", exc)
        log.totals = TokenUsage()
    if path is not None:
        try:
            path.write_text(log.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Extraction usage log saved to {path}")
        except Exception as exc:  # noqa: BLE001
            _record_logging_error(log, "log persistence", exc)


@contextmanager
def extraction_usage_log(
    output_dir: Path,
    study_name: str,
    study_type: str,
    model: str,
) -> Generator[ExtractionUsageLog | None, None, None]:
    """Collect usage best-effort without suppressing genuine extraction failures."""
    log = None
    path = None
    stack = ExitStack()
    try:
        log = ExtractionUsageLog(
            study_name=study_name, study_type=study_type, model=model
        )
        path = output_dir / f"extraction_log_{study_name}_{log.run_id}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(log.model_dump_json(indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _record_logging_error(log, "initialization", exc)
    if log is not None:
        try:
            callback = UsageCallback(log)
            stack.enter_context(
                dspy.context(
                    callbacks=[*dspy.settings.callbacks, callback],
                    disable_history=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            _record_logging_error(log, "callback setup", exc)
    try:
        yield log
    except BaseException as exc:
        if log is not None:
            log.status = "failed"
            log.error_type = type(exc).__name__
        raise
    else:
        if log is not None:
            log.status = "completed"
    finally:
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            _record_logging_error(log, "callback cleanup", exc)
        if log is not None:
            _finalize_log(log, path)
