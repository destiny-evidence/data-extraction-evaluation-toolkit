"""Sub-commands for project initialisation wizard."""

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from deet.data_models.project import DeetProject
from deet.utils.cli_utils import echo_and_log, fail_with_message

app = typer.Typer(help="Project-related commands")
console = Console()

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
    echo_and_log(
        f"✅ Default config exported to {DEFAULT_CONFIG_PATH}", fg=typer.colors.GREEN
    )
    echo_and_log(
        "✏️  Edit this file to adjust options for data extraction.", fg=typer.colors.BLUE
    )


@app.command()
def init() -> None:
    """Initialise a new project."""
    try:
        project = DeetProject.load()
        echo_and_log("Warning! Project already exists")
        overwrite = typer.confirm(
            "Would you like to continue?" "This may overwrite data and settings"
        )
        if not overwrite:
            fail_with_message("Exiting..")
    except SystemExit:
        echo_and_log("Creating new project")
    welcome = Panel(
        "[bold cyan]deet Project Initialiser[/]\n\n"
        "Let's collect a few bits of information about your new project.\n"
        "Press Ctrl-C at any time to abort.\n",
        title="🚤  Welcome",
        border_style="bright_blue",
        box=box.ROUNDED,
    )
    console.print(welcome)

    project = DeetProject.init_interactive()

    settings = Panel(
        "[bold cyan] Success![/]\n\n" "Now let us configure your settings",
        title="✅  Project successfully set up!",
        border_style="bright_blue",
        box=box.ROUNDED,
    )

    console.print(settings)
    project.populate_env()

    echo_and_log("Writing default config file")
    export_config_template()

    processed_data = project.process_data()
    echo_and_log("Successfully parsed processed data.")

    echo_and_log("Initialising prompt definition file.")
    processed_data.export_attributes_csv_file(filepath=project.prompt_csv_path)

    echo_and_log("Initialising reference-pdf link mapping file.")
    processed_data.export_linkage_mapper_csv(file_path=project.link_map_path)

    project.dump_to_toml()
    settings = Panel(
        "[bold green] Success![/]\n\n" "Your project is now ready to use",
        title="✅  Project successfully set up!",
        border_style="bright_blue",
        box=box.ROUNDED,
    )
