"""
Data models for DeetProject.

DeetProjects handle the one-time definition of configuration options,
and create standardised directory structures to store resources like
prompt csvs, link maps, experiment results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
    Field,
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
            valid="Must be at least 1 character",
        ),
    ] = Field(..., description="The name of a deet project", min_length=1)

    gold_standard_data_format: Annotated[
        SupportedImportFormat,
        UI(
            help=(
                "The format of a file describing documents,"
                " attributes, and annotations."
            )
        ),
    ] = Field(..., description="Format of gold standard annotations")

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
    ] = Field(..., description="Path to gold standard annotated data")

    pdf_dir: Annotated[
        Path | None,
        UI(
            help=(
                "If you want to extract data from full texts, "
                "choose a directory that contains your pdfs."
                " You will have an opportunity to link this later"
                " using a 'link map' created here"
            )
        ),
    ] = Field(None, description="Path to folder containing PDFs")

    # Project metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Root defaults to cwd
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
    def gold_standard_data_abspath(self) -> Path:
        """
        Return a usable path to the gold-standard data.

        ``gold_standard_data_path`` is stored relative to the project root (and
        never persisted as an absolute path, so ``project.yaml`` stays portable).
        This joins it with the root only for I/O.
        """
        return self.root / self.gold_standard_data_path

    @property
    def pdf_dir_abspath(self) -> Path | None:
        """
        Return a usable path to the pdf directory, or None if unset.

        ``pdf_dir`` is stored relative to the project root and joined with it here
        only for I/O; it is never persisted as an absolute path.
        """
        if self.pdf_dir is None:
            return None
        return self.root / self.pdf_dir

    # Configuration and validation
    model_config = ConfigDict(
        json_encoders={Path: str},
        extra="ignore",
    )

    @field_validator("gold_standard_data_path", mode="after")
    @classmethod
    def check_suffix(cls, value: Path) -> Path:
        """Check if extension is supported."""
        if value.suffix not in SUPPORTED_EXTENSIONS:
            unsupported_ext = f"Unsupported extension, allowed: {SUPPORTED_EXTENSIONS}"
            raise ValueError(unsupported_ext)
        return value

    @field_validator("pdf_dir", mode="before")
    @classmethod
    def _process_pdf_dir(cls, value: object) -> object | None:
        """Parse empty string to None (not cwd) before Path coercion."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    def anchor_to(self, root: Path, source_dir: Path | None = None) -> None:
        """
        Anchor the project to ``root``, re-expressing resource paths relative to it.

        Resource paths are authored relative to ``source_dir`` (default cwd, i.e.
        where the wizard ran). They are rewritten relative to ``root`` so they stay
        correct when ``root`` differs from that directory (as with
        ``deet project new``). The stored values remain relative, never absolute.
        """
        source = source_dir or Path.cwd()
        self.gold_standard_data_path = (
            source / self.gold_standard_data_path
        ).relative_to(root, walk_up=True)
        if self.pdf_dir is not None:
            self.pdf_dir = (source / self.pdf_dir).relative_to(root, walk_up=True)
        self._root = root

    def validate_resources(self) -> None:
        """Check that the project's resource paths exist on disk."""
        if not self.gold_standard_data_abspath.exists():
            missing = (
                f"Gold standard data not found at {self.gold_standard_data_abspath} "
                f"(stored as '{self.gold_standard_data_path}', relative to {self.root})"
            )
            raise FileNotFoundError(missing)
        if self.pdf_dir_abspath is not None and not self.pdf_dir_abspath.is_dir():
            missing_pdfs = (
                f"PDF directory not found at {self.pdf_dir_abspath} "
                f"(stored as '{self.pdf_dir}', relative to {self.root})"
            )
            raise FileNotFoundError(missing_pdfs)

    def setup(self) -> None:
        """
        Set a project up.

        Create directory structure, process gold-standard data, and create
            prompt csv and link map
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self.validate_resources()

        processed_data = self.process_data()
        notify("Successfully parsed processed data.", level=LogLevel.SUCCESS)

        processed_data.export_attributes_csv_file(filepath=self.prompt_csv_path)
        notify("Initialised prompt definition file.", level=LogLevel.SUCCESS)

        processed_data.export_linkage_mapper_csv(
            file_path=self.link_map_path, document_base_dir=self.pdf_dir_abspath
        )
        notify("Initialised reference-pdf link mapping file.", level=LogLevel.SUCCESS)

        self.export_config_template()
        notify("Exported default config template", level=LogLevel.SUCCESS)

        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.linked_documents_path.mkdir(parents=True, exist_ok=True)

        self.dump_to_yaml()

    def dump_to_yaml(self) -> None:
        """
        Write a minimal ``project.yaml`` file to save project options.

        Written to the project root by default. Resource paths are stored as their
        relative values, never resolved to absolute, so the file stays portable.
        """
        target = self.root / PROJECT_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
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
    def load(cls, project_dir: Path | None = None) -> DeetProject:
        """
        Load the project from ``project_dir`` (default: the current directory).

        The project root is anchored to that directory; stored resource paths stay
        relative and are resolved against it. deet commands are run from the project
        directory.
        """
        root = (project_dir or Path.cwd()).resolve()
        project_file = root / PROJECT_FILE
        if not project_file.is_file():
            not_found = f"No {PROJECT_FILE} found in {root}"
            raise FileNotFoundError(not_found)
        data = yaml.safe_load(project_file.read_text())
        project = cls.model_validate(data["project"])
        project._root = root  # noqa: SLF001
        return project

    def process_data(self) -> ProcessedAnnotationData:
        """Process the project's gold standard data."""
        converter = self.gold_standard_data_format.get_annotation_converter()
        return converter.process_annotation_file(self.gold_standard_data_abspath)


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

    @property
    def llm_annotation_csv(self) -> Path:
        """Return location of csv containing llm extractions."""
        return self.base_dir / "llm_annotations.csv"

    @property
    def extraction_metadata(self) -> Path:
        """Return path to extraction metadata JSON (cost, tokens, timing)."""
        return self.base_dir / "extraction_metadata.json"
