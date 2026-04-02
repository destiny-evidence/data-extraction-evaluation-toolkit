"""Sub-commands for project initialisation wizard."""

from pathlib import Path

import typer
from InquirerPy import inquirer
from rich import box
from rich.panel import Panel

from deet.data_models.project import DeetProject
from deet.settings import LogLevel
from deet.ui import console, notify
from deet.ui.wizards import run_model_wizard
from deet.utils.cli_utils import fail_with_message

app = typer.Typer(help="Project-related commands")

DEFAULT_CONFIG_PATH = Path("default_extraction_config.yaml")


def export_config_template() -> None:
    """Export the default DataExtractionConfig to a YAML file."""
    import yaml  # type:ignore[import-untyped]

    from deet.extractors.llm_data_extractor import DataExtractionConfig

    config = DataExtractionConfig()
    DEFAULT_CONFIG_PATH.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    notify(
        f"✅ Default config exported to {DEFAULT_CONFIG_PATH}", level=LogLevel.SUCCESS
    )
    notify(
        "✏️  Edit this file to adjust options for data extraction.", level=LogLevel.INFO
    )


@app.command()
def init() -> None:
    """Initialise a new project."""
    if DeetProject.exists():
        project = DeetProject.load()
        notify(
            (
                "A Project already exists in this directory. "
                "Continuing could overwrite data and settings"
            ),
            level=LogLevel.WARNING,
        )
        if not inquirer.confirm("Overwrite existing project?").execute():
            fail_with_message("Exiting..")

    console.clear()
    welcome = Panel(
        "[bold cyan]deet Project Initialiser[/]\n\n"
        "Let's collect a few bits of information about your new project.\n"
        "Press Ctrl-C at any time to abort.\n",
        title="🚤  Welcome",
        border_style="bright_blue",
        box=box.DOUBLE,
    )
    console.print(welcome)

    project = run_model_wizard(DeetProject)
    project.setup()

    console.clear()
    success = Panel(
        "[bold green] Success![/]\n\n" "Your project is now ready to use",
        title="✅  Project successfully set up!",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(success)

    settings = Panel(
        "[bold cyan] Configuration[/]\n\n" "Now let us configure your settings",
        title="⚙️  Project settings",
        border_style="bright_blue",
        box=box.ROUNDED,
    )

    console.print(settings)
    project.populate_env()

    export_config_template()
