"""App settings using pydantic-settings."""

from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from dotenv import set_key
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from deet.data_models.ui_schema import UI


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


class LogLevel(StrEnum):
    """Supported log levels for logging."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    log_level: LogLevel = Field(
        default=LogLevel.DEBUG,
        description="log level for the app logger.",
        json_schema_extra={"skip_prompt": True},
    )

    runtime: Runtime = Field(
        default=Runtime.LOCAL,
        description="Runtime environment.",
        json_schema_extra={"skip_prompt": True},
    )

    # Provider credentials / settings (secrets redacted)
    azure_api_key: Annotated[
        SecretStr | None, UI(help="Press enter to leave this unchanged")
    ] = Field(
        default=None,
        description="Azure OpenAI API key if using Azure provider.",
    )
    azure_api_base: Annotated[
        SecretStr | None, UI(help="Press enter to leave this unchanged")
    ] = Field(default=None, description="Base URL for azure openAI.")

    # disk cache folder
    base_disk_cache_dir: Path = Field(
        default=(Path.home() / ".deet_cache"),
        description="the base directory for disk-based caches.",
        json_schema_extra={"skip_prompt": True},
    )

    def dump_to_env(self, target_path: Path = Path(".env")) -> None:
        """Serialise settings object to a .env file."""
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not target_path.exists():
            target_path.touch()

        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, SecretStr):
                    str_value = value.get_secret_value()
                else:
                    str_value = str(value)

                set_key(
                    str(target_path), field_name.upper(), str_value, quote_mode="always"
                )


@lru_cache(maxsize=1)
def get_settings() -> DataExtractionSettings:
    """Return a cached settings instance for reuse across the process."""
    return DataExtractionSettings()
