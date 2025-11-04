"""API config parsing and model."""

from enum import StrEnum
from functools import lru_cache

from pydantic import UUID4, EmailStr, Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """
    Environment that the Fetch Abstracts Robot is running in.

    **Allowed values**:
    - `local`: The robot is running locally
    - `development`: The robot is running in development
    - `staging`: The robot is running in staging
    - `test`: The robot is running as a test fixture for the repository
    - `production`: The robot is running in production
    """

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Settings model for API."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    robot_title: str = Field(
        default="Fetch Abstracts Robot (FAR)",
        description="The title of the robot.",
    )
    robot_secret: str | None = Field(
        default=None,
        description="Secret needed for communicating with destiny repo.",
    )
    robot_id: UUID4 | None = Field(
        default=None,
        description="Client id needed for communicating with destiny repository.",
    )
    destiny_repository_url: HttpUrl

    mailto: EmailStr | None = Field(
        default="test@test.com",
        description="mailto param to add to crossref req headers",
    )

    env: Environment = Field(
        default=Environment.STAGING,
        description="The environment the robot is deployed in.",
    )

    poll_interval_seconds: int = Field(
        default=30,
        description=("How often to poll for new robot enhancement batches (seconds)"),
    )

    batch_size: int = Field(
        default=2,
        description=("The number of references to include per enhancement batch"),
    )

    # API keys for abstract retrieval
    elsevier_scopus_key: SecretStr | None = Field(
        default=None, description="api key for elsevier scopus api."
    )
    elsevier_scopus_inst_token: SecretStr | None = Field(
        default=None, description="inst token for elsevier scopus api."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get a cached settings object."""
    return Settings()  # type: ignore[call-arg]
