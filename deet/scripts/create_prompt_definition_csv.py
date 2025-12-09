"""
Command line script to create a csv in a project
(borrowing from pipeline_prompt_csv_01.py).
"""

import typer

from deet.data_models.project import DeetProject
from deet.processors.eppi_annotation_converter import EppiAnnotationConverter

app = typer.Typer(help="Create a csv of prompt definitions from your proc_data")


converter = EppiAnnotationConverter()


@app.command()
def create_prompt_definition_csv() -> None:
    """Create a csv of prompt definitions from your processed data."""
    proj = DeetProject(path=".")
    csv_path = proj.prompt_folder / "definitions.csv"
    for attribute in proj.read_attributes():
        attribute.write_to_csv(csv_path)


def main() -> None:
    """Run the typer app."""
    app()


if __name__ == "__main__":
    main()
