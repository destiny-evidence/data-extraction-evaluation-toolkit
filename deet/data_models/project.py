"""
Data models for DeetProject.
Handles the one-time definition of configuration options.
"""

from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

import typer
from dotenv import set_key
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.data_models.ui_schema import UI
from deet.processors.converter_register import SupportedImportFormat
from deet.settings import DataExtractionSettings, LogLevel
from deet.ui import notify
from deet.utils.cli_utils import inquire_pydantic_field

PROJECT_FILE = Path("project.toml")


class EnvironmentFile(StrEnum):
    """A choice of where to store settings."""

    PROJECT = auto()
    SYSTEM = auto()

    def env_file(self) -> Path:
        """Map option to env file."""
        mapping = {
            EnvironmentFile.PROJECT: Path(".env"),
            EnvironmentFile.SYSTEM: Path.home() / ".deet" / ".env",
        }
        return mapping[self]


class DeetProject(BaseModel):
    """
    A deet "project" that lives in a directory.
    Configuration options are defined here once, and elicited through an
        interactive wizard.
    """

    name: Annotated[
        str,
        UI(
            help="Give your project a name. This will help you to identify it later",
            valid="Must be at least 2 characters",
        ),
    ] = Field(..., description="The name of a deet project", min_length=2)
    created_at: datetime = datetime.now(UTC)
    gold_standard_data_path: Annotated[
        Path,
        UI(
            help=(
                "A file containing a list of documents from which you wish to"
                " extract data"
                ", and (optionally) a set of human annotations to be used"
                " to evaluate "
                "automatic extraction."
            ),
            instructions="press Tab to autocomplete, '/' to go to next directory",
            valid="Must be a valid .csv or .json path",
        ),
    ] = Field(..., description="Path to raw data")

    gold_standard_data_format: Annotated[
        SupportedImportFormat,
        UI(
            help=(
                "The format of your raw data. "
                "Choose from the list of supported formats"
            )
        ),
    ] = Field(..., description="Format of gold standard annotations")

    environment_file: Annotated[
        EnvironmentFile,
        UI(
            help=(
                "Where to store your API keys."
                " Select system if you want to re-use credentials across deet projects,"
                " or Project if you wish to use specific credentials for this project"
            )
        ),
    ] = Field(default=EnvironmentFile.SYSTEM, description="Environment file")

    pdf_dir: Annotated[
        Path | None,
        UI(
            help=(
                "If you want to extract data from full texts, "
                "choose a directory that contains your pdfs."
                " You will have an opportunity to link this later"
                " by running `deet link-documents-fulltexts`"
            )
        ),
    ] = Field(None, description="Path to folder containing PDFs")

    out_dir: Path = Field(default=Path("data-extraction-experiments/"))
    prompt_csv_path: Path = Path("prompts/prompt_definitions.csv")
    link_map_path: Path = Path("link_map.csv")
    linked_documents_path: Path = Path("linked_documents")

    model_config = ConfigDict(
        json_encoders={Path: str},
        extra="ignore",
    )

    def setup(self) -> None:
        """
        Set a project up.

        Create directory structure, process gold-standard data, and create
            prompt csv and link map
        """
        self.create_directory_structure()
        processed_data = self.process_data()
        notify("Successfully parsed processed data.", level=LogLevel.SUCCESS)

        processed_data.export_attributes_csv_file(filepath=self.prompt_csv_path)
        notify("Initialised prompt definition file.", level=LogLevel.SUCCESS)

        processed_data.export_linkage_mapper_csv(file_path=self.link_map_path)
        notify("Initialised reference-pdf link mapping file.", level=LogLevel.SUCCESS)

        self.dump_to_toml()

    @field_validator("gold_standard_data_path", mode="before")
    @classmethod
    def _abs_and_check_exists(cls, value: str | Path) -> Path:
        p = Path(value) if not isinstance(value, Path) else value
        abs_path = p.resolve()
        if not abs_path.is_file():
            no_gs = f"Gold-standard file does not exist: {abs_path}"
            raise ValueError(no_gs)
        return abs_path

    @field_validator("pdf_dir", mode="before")
    @classmethod
    def _process_pdf_dir(cls, value: str | Path) -> Path | None:
        if value == "":
            return None
        p = Path(value) if not isinstance(value, Path) else value
        return p.resolve()

    def dump_to_toml(self, target: Path = PROJECT_FILE) -> None:
        """Write a minimal ``project.toml`` file to save project options."""
        import toml

        data = {"project": self.model_dump(mode="json")}
        with target.open("w", encoding="utf-8") as f:
            toml.dump(data, f)

    def create_directory_structure(self) -> None:
        """Create necessary directories for project."""
        Path("./prompts").mkdir(exist_ok=True)
        logger.info("creating directories")

    @classmethod
    def exists(cls) -> bool:
        """Check if project exists in current directory."""
        return PROJECT_FILE.exists()

    @classmethod
    def load(cls, filename: Path = PROJECT_FILE) -> "DeetProject":
        """Load a project from a toml file."""
        import toml

        try:
            data = toml.load(filename.open())
        except FileNotFoundError as err:
            import sys

            sys.tracebacklimit = -1
            no_project = "This directory doesn't contain a deet project."
            raise SystemExit(no_project) from err
        return cls.model_validate(data["project"])

    def populate_env(self) -> None:
        """Populate environment file."""
        target = self.environment_file.env_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            target.touch()
            write_file = True
        else:
            write_file = typer.confirm(
                f"{self.environment_file} env file already exists."
                " Would you like to overwrite it?"
            )

        if write_file:
            for name, field in DataExtractionSettings.model_fields.items():
                answer = inquire_pydantic_field(field)
                if answer:
                    set_key(target, name, answer, quote_mode="always")

    def process_data(self) -> ProcessedAnnotationData:
        """Process the project's gold standard data."""
        converter = self.gold_standard_data_format.get_annotation_converter()
        return converter.process_annotation_file(self.gold_standard_data_path)
