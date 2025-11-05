"""App settings using pydantic-settings."""

import os
from collections.abc import Callable
from enum import StrEnum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Runtime(StrEnum):
    """Permitted runtime environments for DEET."""

    LOCAL = auto()
    DEVELOPMENT = auto()
    STAGING = auto()
    TEST = auto()
    PRODUCTION = auto()


class DataExtractionSettings(BaseSettings):
    """
    Settings model for data extraction behavior and provider credentials.

    All fields are fully typed and documented. Unknown environment variables
    are forbidden to help catch configuration drift early.
    """

    model_config = SettingsConfigDict(
        extra="forbid",
    )

    # General
    env: Runtime = Field(
        default=Runtime.LOCAL,
        description="Runtime environment.",
    )

    # LLM configuration
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier used for completions.",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for the LLM.",
        ge=0.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum number of tokens to generate (None means provider default)."
        ),
    )

    # Context and selection
    context_type: str = Field(
        default="full_document",
        description=(
            "How to prepare context for the LLM. One of: full_document, abstract_only, "
            "rag_snippets, custom."
        ),
    )
    max_context_length: int = Field(
        default=40000,
        description="Maximum length of prepared context (characters).",
        # TO DO: turn this into tokens; not characters
        ge=1,
    )
    selected_attribute_ids: list[str] = Field(
        default_factory=list,
        description="Filter for specific attribute IDs to extract.",
    )

    # Output toggles
    include_reasoning: bool = Field(
        default=True,
        description="Include model reasoning in the output structure.",
    )
    include_additional_text: bool = Field(
        default=True,
        description="Include additional text/citations in the output structure.",
    )

    # Provider credentials / settings (secrets redacted)
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key if using OpenAI provider.",
    )
    azure_api_key: SecretStr | None = Field(
        default=None,
        description="Azure OpenAI API key if using Azure provider.",
    )
    azure_deployment: str | None = Field(
        default=None,
        description="Azure deployment name to use when azure_api_key is provided.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        _settings_cls: type[BaseSettings],
        init_settings: Callable[..., dict[str, Any]],
        env_settings: Callable[..., dict[str, Any]],
        _dotenv_settings: Callable[..., dict[str, Any]],
        file_secret_settings: Callable[..., dict[str, Any]],
    ) -> tuple[Callable[..., dict[str, Any]], ...]:
        """
        Customize sources to prefer YAML file over env and defaults.

        Order: YAML file -> environment variables -> init kwargs -> file secrets
        """

        def yaml_settings_source(_: BaseSettings) -> dict[str, Any]:
            # Determine config path: env var or default_config.yaml at repo root
            config_path_str = os.getenv("DATA_EXTRACTION_CONFIG_FILE")
            if config_path_str:
                cfg_path = Path(config_path_str)
            else:
                # default to repo root / default_config.yaml
                repo_root = Path(__file__).resolve().parents[1]
                cfg_path = repo_root / "default_config.yaml"

            if not cfg_path.exists():
                return {}
            try:
                with cfg_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    return data
                return {}  # noqa: TRY300
            except (OSError, ValueError, yaml.YAMLError):
                # Fail silent to allow env/defaults to take over
                return {}

        return (
            yaml_settings_source,
            env_settings,
            init_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> DataExtractionSettings:
    """Return a cached settings instance for reuse across the process."""
    return DataExtractionSettings()
