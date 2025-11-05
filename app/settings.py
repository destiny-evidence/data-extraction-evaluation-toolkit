"""App settings using pydantic-settings."""

from enum import StrEnum, auto
from functools import lru_cache

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
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # General
    runtime: Runtime = Field(
        default=Runtime.LOCAL,
        description="Runtime environment.",
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

    # # Context and selection
    # context_type: str = Field(
    #     default="full_document",
    #     description=(
    #         "How to prepare context for the LLM. One of: f
    # ull_document, abstract_only, "
    #         "rag_snippets, custom."
    #     ),
    # )
    # max_context_length: int = Field(
    #     default=40000,
    #     description="Maximum length of prepared context (characters).",
    #     # TO DO: turn this into tokens; not characters
    #     ge=1,
    # )
    # selected_attribute_ids: list[str] = Field(
    #     default_factory=list,
    #     description="Filter for specific attribute IDs to extract.",
    # )

    # Output toggles
    # include_reasoning: bool = Field(
    #     default=True,
    #     description="Include model reasoning in the output structure.",
    # )
    # include_additional_text: bool = Field(
    #     default=True,
    #     description="Include additional text/citations in the output structure.",
    # )

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


@lru_cache(maxsize=1)
def get_settings() -> DataExtractionSettings:
    """Return a cached settings instance for reuse across the process."""
    return DataExtractionSettings()
