"""Command line script to create a project."""

import shutil
from pathlib import Path

import typer
import yaml
from loguru import logger

from deet.data_models.project import DeetProject
from deet.extractors.llm_data_extractor import DataExtractionConfig
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter

app = typer.Typer(help="Create a DEET project")


@app.command()
def create_project(path: str, data_path: str, *, clear: bool = False) -> None:
    """Set up a folder structure for a DEET project."""
    if path == ".":
        typer.echo("creating project in current directory")
    else:
        typer.echo(f"creating project in {path}")

    proj = DeetProject(path=path)
    if (
        proj.p.exists()
        and clear
        and not typer.confirm(
            "This project already exists. Are you sure you "
            "want to clear the data in it?"
        )
    ):
        raise typer.Abort()  # noqa: RSE102

    for f in proj.folders:
        if clear:
            shutil.rmtree(f)
        f.mkdir(exist_ok=not clear)

    input_path = proj.raw_data / "data.json"
    with Path(data_path).open() as src:
        input_path.write_text(src.read())

    # Create config file with defaults
    config = DataExtractionConfig()
    with proj.p.joinpath("run-settings.yaml").open("w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    convert_raw_data(input_path, proj.proc_data)


def convert_raw_data(input_path: Path, output_path: Path) -> None:
    """Process EPPIJson file."""
    converter = EppiAnnotationConverter()
    processed_data = converter.process_annotation_file(str(input_path))
    saved_files = converter.save_processed_data(processed_data, str(output_path))

    logger.info("Conversion complete!")
    logger.info(output_path)
    logger.info(f"Files saved to: {Path(saved_files['attributes']).parent.absolute()}")
    for file_type, file_path in saved_files.items():
        logger.info(f"  {file_type}: {Path(file_path).absolute()}")


def main() -> None:
    """Run the create project cli app."""
    app()


if __name__ == "__main__":
    main()
