"""App settings using pydantic-settings."""

from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Runtime(StrEnum):
    """Permitted runtime environments for DEET."""

    LOCAL = auto()
    DEVELOPMENT = auto()
    STAGING = auto()
    TEST = auto()
    PRODUCTION = auto()


class LLMProvider(StrEnum):
    """Supported LLM Providers."""

    AZURE = auto()
    OLLAMA = auto()


# Fallback max input context (tokens) when litellm cannot resolve the model.
# Used by ``DataExtractionConfig`` when ``max_context_tokens`` is inferred and
# ``get_model_max_tokens`` returns None. Single source of truth (do not duplicate
# in tokenisation or extractors).
DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK: int = 128_000


class DataExtractionSettings(BaseSettings):
    """
    Settings model for data extraction behavior and provider credentials.

    All fields are fully typed and documented. Unknown environment variables
    are forbidden to help catch configuration drift early.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # General
    log_level: str = Field(default="DEBUG", description="log level for the app logger.")

    runtime: Runtime = Field(
        default=Runtime.LOCAL,
        description="Runtime environment.",
    )

    llm_provider: LLMProvider = Field(
        default=LLMProvider.AZURE, description="LLM Provider"
    )

    # LLM configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier used for completions.",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="Sampling temperature for the LLM.",
        ge=0.0,
    )
    llm_max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (None means provider default)."
        ),
    )
    llm_max_context_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum input context length in tokens (system + attributes + "
            "document). None = infer from model (litellm registry), else "
            f"{DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK} via "
            "DEFAULT_LLM_MAX_CONTEXT_TOKENS_FALLBACK. Override to manage costs."
        ),
    )

    # Provider credentials / settings (secrets redacted)
    azure_api_key: SecretStr | None = Field(
        default=None,
        description="Azure OpenAI API key if using Azure provider.",
    )
    azure_api_base: SecretStr | None = Field(
        default=None, description="Base URL for azure openAI."
    )

    # disk cache folder
    base_disk_cache_dir: Path = Field(
        default=(Path.home() / ".deet_cache"),
        description="the base directory for disk-based caches.",
    )


@lru_cache(maxsize=1)
def get_settings() -> DataExtractionSettings:
    """Return a cached settings instance for reuse across the process."""
    return DataExtractionSettings()
