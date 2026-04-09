"""
Data models for DeetProject.
Handles the one-time definition of configuration options.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from deet.data_models.processed_gold_standard_annotations import (
        ProcessedAnnotationData,
    )


import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    FilePath,
    PrivateAttr,
    field_validator,
)

from deet.data_models.ui_schema import UI
from deet.processors.converter_register import (
    SUPPORTED_EXTENSIONS,
    SupportedImportFormat,
)
from deet.settings import LogLevel
from deet.ui import notify

PROJECT_FILE = Path("project.yaml")


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

    # Wizard fields, configurable by users
    name: Annotated[
        str,
        UI(
            help="Give your project a name. This will help you to identify it later",
            valid="Must be at least 2 characters",
        ),
    ] = Field(..., description="The name of a deet project", min_length=2)

    gold_standard_data_path: Annotated[
        FilePath,
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
        DirectoryPath | None,
        UI(
            help=(
                "If you want to extract data from full texts, "
                "choose a directory that contains your pdfs."
                " You will have an opportunity to link this later"
                " by running `deet link-documents-fulltexts`"
            )
        ),
    ] = Field(None, description="Path to folder containing PDFs")

    # Project metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Root is set during load() or defaults to cwd
    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def root(self) -> Path:
        """Return project root."""
        return self._root

    # Computed paths - file and folder structure within project dir
    @property
    def experiments_dir(self) -> Path:
        """
        Return path to experiments directory.

        Each time we run data extraction in this project, the results
        of the experiment will be stored here.
        """
        return self.root / "data-extraction-experiments"

    @property
    def prompt_csv_path(self) -> Path:
        """Return path to prompt definition file."""
        return self.root / "prompts" / "prompt_definitions.csv"

    @property
    def link_map_path(self) -> Path:
        """Return path to link map."""
        return self.root / "link_map.csv"

    @property
    def linked_documents_path(self) -> Path:
        """Return path to linked documents folder."""
        return self.root / "linked_documents"

    @property
    def config_path(self) -> Path:
        """Return path to config file."""
        return self.root / "default_extraction_config.yaml"

    @property
    def env_path(self) -> Path:
        """Return path to project .env file."""
        return self.root / ".env"

    # Configuration and validation
    model_config = ConfigDict(
        json_encoders={Path: str},
        extra="ignore",
    )

    @field_validator("gold_standard_data_path", mode="after")
    @classmethod
    def _abs_and_check_suffix(cls, value: Path) -> Path:
        """Return absolute path, and check if extension is supported."""
        abs_path = value.resolve()
        if abs_path.suffix not in SUPPORTED_EXTENSIONS:
            unsupported_ext = f"Unsupported extension, allowed: {SUPPORTED_EXTENSIONS}"
            raise ValueError(unsupported_ext)
        return abs_path

    @field_validator("pdf_dir", mode="after")
    @classmethod
    def _process_pdf_dir(cls, value: Path) -> Path | None:
        """Parse empty string to None (not cwd), otherwise resolve path."""
        if value == "":
            return None
        return value.resolve()

    def setup(self) -> None:
        """
        Set a project up.

        Create directory structure, process gold-standard data, and create
            prompt csv and link map
        """
        processed_data = self.process_data()
        notify("Successfully parsed processed data.", level=LogLevel.SUCCESS)

        processed_data.export_attributes_csv_file(filepath=self.prompt_csv_path)
        notify("Initialised prompt definition file.", level=LogLevel.SUCCESS)

        processed_data.export_linkage_mapper_csv(file_path=self.link_map_path)
        notify("Initialised reference-pdf link mapping file.", level=LogLevel.SUCCESS)

        self.export_config_template()
        notify("Exported default config template", level=LogLevel.SUCCESS)

        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.linked_documents_path.mkdir(parents=True, exist_ok=True)

        self.dump_to_yaml()

    def dump_to_yaml(self, target: Path = PROJECT_FILE) -> None:
        """Write a minimal ``project.yaml`` file to save project options."""
        data = {"project": self.model_dump(mode="json")}
        with target.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

    def export_config_template(self) -> None:
        """Export a default config template."""
        from deet.extractors.llm_data_extractor import DataExtractionConfig

        config = DataExtractionConfig()
        self.config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filename: Path = PROJECT_FILE) -> DeetProject:
        """Load a project from a toml file."""
        data = yaml.safe_load(filename.read_text())
        return cls.model_validate(data["project"])

    @classmethod
    def exists(cls) -> bool:
        """Check if project exists in current directory."""
        return PROJECT_FILE.exists()

    def process_data(self) -> ProcessedAnnotationData:
        """Process the project's gold standard data."""
        converter = self.gold_standard_data_format.get_annotation_converter()
        return converter.process_annotation_file(self.gold_standard_data_path)


@dataclass(frozen=True)
class ExperimentArtefacts:
    """Defines the structure of a data extraction experiment directory."""

    base_dir: Path
    run_id: str

    @property
    def metrics(self) -> Path:
        """Return location of experiment metrics."""
        return self.base_dir / "metrics.csv"

    @property
    def comparison(self) -> Path:
        """Return location of csv comparing goldstandard to llm extractions."""
        return self.base_dir / "goldstandard_llm_comparison.csv"

    @property
    def prompts_snapshot(self) -> Path:
        """Return location of csv capturing prompts used."""
        return self.base_dir / "prompts_used.csv"

    @property
    def config_snapshot(self) -> Path:
        """Return location of csv capturing config used."""
        return self.base_dir / "config.yaml"

    @property
    def llm_annotations(self) -> Path:
        """Return location of json containing llm extractions."""
        return self.base_dir / "llm_annotations.json"
