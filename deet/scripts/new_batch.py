"""Command line script to create a new batch of records in a project."""

import json
from typing import Annotated

import typer

from deet.data_models.project import DeetProject

app = typer.Typer(help="Create a DEET project")


@app.command()
def new_batch(
    n: Annotated[int, typer.Argument()],
) -> None:
    """Create a new batch with `n` documents."""
    proj = DeetProject(path=".")

    if not proj.folders_exist():
        typer.echo(
            "It doesn't seem like you are in a DEET project directory"
            ". Make sure you execute the command from a directory containing"
            "the correct folder structure for DEET projects"
        )
        raise typer.Abort()  # noqa: RSE102

    doc_batch = proj.read_annotated_documents(
        sample=n, exclude=proj.documents_in_batches
    )

    batch_i = len(proj.batches)
    batch_path = proj.batch_folder / f"batch_{batch_i}"
    batch_path.mkdir()

    id_path = batch_path / "batch_ids.json"
    with id_path.open("w") as f:
        json.dump([doc.document_id for doc in doc_batch], f)


def main() -> None:
    """Run the typer app."""
    app()


if __name__ == "__main__":
    main()
