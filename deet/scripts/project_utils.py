# ruff: noqa: PLC0415
"""CLI helpers for the ``deet project`` setup/creation commands."""

from pathlib import Path
from typing import Annotated

import typer
from InquirerPy import inquirer
from pydantic import ValidationError

from deet.processors.converter_register import SupportedImportFormat
from deet.settings import DataExtractionSettings, LogLevel
from deet.ui import fail_with_message, notify
from deet.ui.terminal import (
    console,
    continue_after_key,
    render_template,
    run_model_wizard,
)
from deet.ui.terminal.components import info_panel, wizard_field_help
from deet.ui.terminal.wizards import get_ui_metadata, inquire_pydantic_field


def run_init_wizard(root: Path, name: str) -> None:
    """
    Run the interactive project + credentials wizards and set the project up.

    Prompts for every project field except ``name`` (supplied here), anchors the
    project to ``root`` (re-expressing resource paths relative to it), then writes
    the project structure and credentials into ``root``.
    """
    from deet.data_models.project import DeetProject

    console.clear()
    init_md = render_template("project/init")
    console.print(info_panel(init_md, title=":speedboat: project set-up"))
    continue_after_key()

    project = run_model_wizard(DeetProject, prefill={"name": name})
    project.anchor_to(root)
    project.setup()

    console.clear()
    configure_env_md = render_template("project/configure_env.md")
    console.print(info_panel(configure_env_md, ":key: Credential management"))
    continue_after_key()
    settings = run_model_wizard(DataExtractionSettings)
    settings.dump_to_env(target_path=root / ".env")

    new_directory = root.relative_to(Path.cwd())

    console.clear()
    console.print(
        info_panel(
            render_template(
                "project/success.md", project=project, new_directory=str(new_directory)
            )
        )
    )


def guard_overwrite(target_dir: Path, *, force: bool, interactive: bool) -> None:
    """
    Guard against overwriting an existing project at ``target_dir``.

    Does nothing if ``force`` is set or no project exists there. Otherwise prompts
    to overwrite when running interactively, or exits with guidance when headless
    (where prompting is impossible).
    """
    from deet.data_models.project import PROJECT_FILE, DeetProject

    if force or not (target_dir / PROJECT_FILE).exists():
        return
    if not interactive:
        fail_with_message(
            f"A project already exists at {target_dir}. Use --force to overwrite."
        )
    existing_project = DeetProject.load(target_dir)
    notify(
        (
            f"Project {existing_project.name} already exists in {target_dir}. "
            "Continuing could overwrite data and settings"
        ),
        level=LogLevel.WARNING,
    )
    if not inquirer.confirm("Overwrite existing project?").execute():
        fail_with_message("Exiting..")


def create_project(
    root: Path,
    name: str,
    *,
    data_path: Path | None,
    data_type: SupportedImportFormat,
    pdf_dir: Path | None,
) -> None:
    """
    Create and set a project named ``name`` up at ``root``.

    When resource paths are supplied the project is built headlessly from them;
    otherwise the interactive wizard collects the remaining fields. The
    project is anchored to ``root`` and its directory structure written there.
    """
    from deet.data_models.project import DeetProject

    if any([data_path, pdf_dir]):
        try:
            if data_path is None:
                fail_with_message(
                    "Gold-standard data (--data) is required to create a project"
                    " non-interactively"
                )
            project = DeetProject(
                name=name,
                gold_standard_data_path=data_path,
                gold_standard_data_format=data_type,
                pdf_dir=pdf_dir,
            )
        except ValidationError as e:
            fail_with_message(f"Invalid project configuration:\n{e}")
        project.anchor_to(root)
        project.setup()
    else:
        run_init_wizard(root, name)


def prompt_name() -> str:
    """Prompt for a project name."""
    from deet.data_models.project import DeetProject

    info = DeetProject.model_fields["name"]
    ui = get_ui_metadata(info)
    if ui is None:
        no_ui = "No UI component for name"
        raise ValueError(no_ui)
    ui_help = ui.help + (
        ". The name you enter will be standardised and used to create a directory"
    )
    console.clear()
    console.print(wizard_field_help("name", ui_help))
    return str(inquire_pydantic_field(DeetProject, "name", info, ui))


# Shared options, so `init` and `new` accept identical arguments when
# run without the wizard.
DataPathOption = Annotated[
    Path | None,
    typer.Option(
        "--data",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to your gold standard annotation data",
    ),
]
DataFormatOption = Annotated[
    SupportedImportFormat,
    typer.Option("--format", "-t", help="Format of your gold standard annotated data."),
]
PdfDirOption = Annotated[
    Path | None,
    typer.Option(
        "--pdfs",
        "-p",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="The folder where your pdfs for data extraction are stored.",
    ),
]
ForceOption = Annotated[
    bool,
    typer.Option("--force", "-f", help="Overwrite existing project data."),
]
