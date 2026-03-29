"""
Data models for DeetProject.
Handles the one-time definition of configuration options.
"""

from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from deet.data_models.processed_gold_standard_annotations import ProcessedAnnotationData
from deet.processors.converter_register import SupportedImportFormat

PROJECT_FILE = Path("project.toml")


class DeetProject(BaseModel):
    """
    A deet "project" that lives in a directory.
    Configuration options are defined here once, and elicited through an
        interactive wizard.
    """

    name: str
    created_at: datetime = datetime.now(UTC)
    project_path: Path = Path("project.json")
    gold_standard_data_path: Path = Field(
        ..., description="Absolute path to source data"
    )
    gold_standard_data_format: SupportedImportFormat = Field(..., description="Format")
    pdf_dir: Path | None = Field(..., description="Path to folder containing PDFs")
    out_dir: Path = Field(default=Path("data-extraction-experiments/"))
    prompt_csv_path: Path = Path("prompt_definitions.csv")
    link_map_path: Path = Path("link_map.csv")
    linked_documents_path: Path = Path("linked_documents")

    @field_validator("gold_standard_data_path", mode="before")
    def _abs_and_check_exists(self, value: str | Path) -> Path:
        p = Path(value) if not isinstance(value, Path) else value
        abs_path = p.resolve()
        if not abs_path.is_file():
            no_gs = f"Gold-standard file does not exist: {abs_path}"
            raise ValueError(no_gs)
        return abs_path

    @field_validator("pdf_dir", mode="before")
    def _process_pdf_dir(self, value: str | Path) -> Path | None:
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
        logger.info("creating directories")

    @classmethod
    def init_interactive(cls) -> "DeetProject":
        """Initialise a project interactively."""
        from InquirerPy import inquirer

        data = {}

        data["name"] = inquirer.text(
            "Project name:",
            validate=lambda result: len(result.strip()) > 0,
            filter=lambda result: result.strip(),
        ).execute()

        data["gold_standard_data_path"] = inquirer.filepath(
            message="📁  Path to the raw data file (press tab for autocompletion):",
            default="../../taxonomy-DEET/PIK-HIC-mini_v3.json",
            validate=lambda p: Path(p).is_file()
            and Path(p).suffix in [".csv", ".json"],
            invalid_message="Must be valid file .json or .csv file",
        ).execute()

        data["gold_standard_data_format"] = inquirer.select(
            message="Format of the raw data file", choices=list(SupportedImportFormat)
        ).execute()

        data["pdf_dir"] = inquirer.filepath(
            message="📁  Path to a folder containing pdfs:",
            validate=lambda p: Path(p).is_dir(),
            invalid_message="Must be valid file folder",
        ).execute()

        return cls.model_validate(data)

    @classmethod
    def load(cls, filename: Path = PROJECT_FILE) -> "DeetProject":
        """Load a project from a toml file."""
        return cls.model_validate_json(filename.read_text())

    def process_data(self) -> ProcessedAnnotationData:
        """Process the project's gold standard data."""
        converter = self.gold_standard_data_format.get_annotation_converter()
        return converter.process_annotation_file(self.gold_standard_data_path)
