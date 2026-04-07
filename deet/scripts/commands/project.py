"""Sub-commands for project initialisation and configuration."""

from pathlib import Path
from typing import Annotated

import typer
from InquirerPy import inquirer

from deet.data_models.project import DeetProject
from deet.scripts.context import project_required
from deet.settings import DataExtractionSettings, LogLevel
from deet.ui import fail_with_message, notify
from deet.ui.terminal import (
    console,
    continue_after_key,
    render_template,
    run_model_wizard,
)
from deet.ui.terminal.components import info_panel

app = typer.Typer(help="Project-related commands")


@app.command()
def init(ctx: typer.Context) -> None:
    """Initialise a new project."""
    existing_project: DeetProject = ctx.obj.project
    if existing_project is not None:
        notify(
            (
                f"Project {existing_project.name} already exists in this directory. "
                "Continuing could overwrite data and settings"
            ),
            level=LogLevel.WARNING,
        )
        if not inquirer.confirm("Overwrite existing project?").execute():
            fail_with_message("Exiting..")

    console.clear()
    init_md = render_template("project/init")
    console.print(info_panel(init_md, title=":speedboat: project set-up"))
    continue_after_key()

    project = run_model_wizard(DeetProject)
    project.setup()

    console.clear()
    configure_env_md = render_template("project/configure_env.md")
    console.print(info_panel(configure_env_md, ":key: Credential management"))
    continue_after_key()
    settings = run_model_wizard(DataExtractionSettings)
    settings.dump_to_env(project.env_path)

    console.clear()
    console.print(info_panel(render_template("project/success.md", project=project)))


@app.command()
@project_required
def link(ctx: typer.Context) -> None:
    """
    Link documents to their fulltexts.

    This creates a document containing the parsed output of its corresponding
    fulltext in the folder defined in `output_path`. Linking will be
    attempted using a mapping file, if provided, then by matching the
    filename with author and year, then by matching by document id. See
    `deet.processors.linker` for more details.

    """
    from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy

    deet_project: DeetProject = ctx.obj.project
    processed_annotation_data = deet_project.process_data()

    linker = DocumentReferenceLinker(
        references=processed_annotation_data.documents,
        document_base_dir=deet_project.pdf_dir,
        document_reference_mapping=deet_project.link_map_path,
        linking_strategies=[LinkingStrategy.MAPPING_FILE],
    )
    linked_documents = linker.link_many_references_parsed_documents()

    if not deet_project.linked_documents_path.exists():
        deet_project.linked_documents_path.mkdir()

    if len(linked_documents) == 0:
        fail_with_message("Error. Could not link any documents!")

    for linked_document in linked_documents:
        file_path = (
            deet_project.linked_documents_path
            / f"{linked_document.safe_identity.document_id}.json"
        )
        linked_document.save(file_path)


@app.command()
def test_llm_config(
    ctx: typer.Context,
    config_path: Annotated[
        Path | None,
        typer.Option(
            help="A path to a config file containing options for data "
            "extraction config. Leave empty to test the project config."
        ),
    ] = None,
) -> None:
    """Test llm config."""
    from deet.data_models.base import Attribute, AttributeType
    from deet.extractors.cli_helpers import (
        load_config_from_context,
    )
    from deet.extractors.llm_data_extractor import LLMDataExtractor

    config = load_config_from_context(ctx, config_path)
    data_extractor = LLMDataExtractor(config=config)
    attr = Attribute(
        output_data_type=AttributeType.BOOL,
        attribute_id=1234,
        attribute_label="Test Attribute",
        prompt="Is the document about climate and health? Return a BOOL",
    )
    context = (
        "This is document, extract data from me please. I am about climate and health"
    )
    response = data_extractor.extract_from_document(
        attributes=[attr],
        payload=context,
        context_type=None,
    )
    if response.annotations:
        notify(
            (
                f"Successfully returned {len(response.annotations)} annotation: "
                f"{response.annotations}"
            ),
            level=LogLevel.SUCCESS,
        )
