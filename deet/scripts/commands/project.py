# ruff: noqa: PLC0415
"""Sub-commands for project initialisation and configuration."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from deet.data_models.project import DeetProject


import typer
from InquirerPy import inquirer

from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.project_utils import (
    DataFormatOption,
    DataPathOption,
    ForceOption,
    PdfDirOption,
    create_project,
    guard_overwrite,
    prompt_name,
    run_edit,
)
from deet.scripts.typer_context import project_required
from deet.settings import LogLevel
from deet.ui import fail_with_message, notify
from deet.ui.terminal import console, render_template
from deet.ui.terminal.components import info_panel
from deet.utils.text import slugify

app = typer.Typer(help="Commands to create and configure deet projects.")


@app.command()
def init(
    *,
    data_path: DataPathOption = None,
    data_type: DataFormatOption = SupportedImportFormat.EPPI_JSON,
    pdf_dir: PdfDirOption = None,
    force_overwrite: ForceOption = False,
) -> None:
    """
    Initialise a new project in the current directory.

    The project name is taken from the current directory. Leave the data and pdf
    options empty to enter the interactive wizard.
    Use deet project new to create a project in a new directory.
    """
    root = Path.cwd()
    guard_overwrite(
        root, force=force_overwrite, interactive=not any([data_path, pdf_dir])
    )
    create_project(
        root, root.name, data_path=data_path, data_type=data_type, pdf_dir=pdf_dir
    )


@app.command()
def new(
    *,
    name: Annotated[
        str | None, typer.Option(help="Project name (prompted if omitted).")
    ] = None,
    data_path: DataPathOption = None,
    data_type: DataFormatOption = SupportedImportFormat.EPPI_JSON,
    pdf_dir: PdfDirOption = None,
    force_overwrite: ForceOption = False,
) -> None:
    """
    Create a new project in its own directory, named after ``name``.

    Leave the data and pdf options
    empty to enter the interactive wizard. Use `deet project init` to turn the
    current directory into a project.
    """
    if name is None:
        name = prompt_name()

    target = Path.cwd() / slugify(name)
    guard_overwrite(
        target, force=force_overwrite, interactive=not any([data_path, pdf_dir])
    )
    target.mkdir(parents=True, exist_ok=True)
    notify(f"Creating project in {target}", level=LogLevel.INFO)
    create_project(
        target, name, data_path=data_path, data_type=data_type, pdf_dir=pdf_dir
    )


@app.command()
@project_required
def edit(
    typer_context: typer.Context,
    field: Annotated[
        str | None,
        typer.Argument(help="Optionally edit a single field instead of all of them."),
    ] = None,
) -> None:
    """
    Edit an existing project's configuration.

    Re-collects the project fields (pre-filled with the current values) and rewrites
    ``project.yaml`` WITHOUT regenerating artefacts (prompt CSV, link map, experiment
    dirs). Pass a field name to edit just that field; the full edit also lets you
    update credentials.
    """
    run_edit(typer_context.obj.project, field)


@app.command()
@project_required
def regenerate_link_map(typer_context: typer.Context) -> None:
    """
    Regenerate a "link map" from a project.

    A link map is created on project.setup(); this re-creates it.
    """
    if not inquirer.confirm(
        "Overwrite existing link map? Make sure you have saved your work."
    ).execute():
        fail_with_message("Exiting..")
    deet_project: DeetProject = typer_context.obj.project
    processed_annotation_data = deet_project.process_data()

    processed_annotation_data.export_linkage_mapper_csv(
        file_path=deet_project.link_map_path,
        document_base_dir=deet_project.pdf_dir_abspath,
    )


@app.command()
@project_required
def regenerate_prompt_csv(typer_context: typer.Context) -> None:
    """
    Regenerate a prompt csv from a project.

    A prompt csv is created on project.setup(); this re-creates it.
    """
    if not inquirer.confirm(
        "Overwrite existing prompt csv? Make sure you have saved your work."
    ).execute():
        fail_with_message("Exiting..")
    deet_project: DeetProject = typer_context.obj.project
    processed_annotation_data = deet_project.process_data()

    processed_annotation_data.export_attributes_csv_file(
        filepath=deet_project.prompt_csv_path
    )


@app.command()
@project_required
def regenerate_config_template(typer_context: typer.Context) -> None:
    """
    Regenerate config template from a project.

    A config template with defaults for each option is created on project.setup();
    this re-creates it.
    """
    if not inquirer.confirm(
        "Overwrite existing config template? Make sure you have saved your work."
    ).execute():
        fail_with_message("Exiting..")
    deet_project: DeetProject = typer_context.obj.project
    deet_project.export_config_template()


@app.command()
@project_required
def link(typer_context: typer.Context) -> None:
    """
    Link documents to their fulltexts.

    This creates a document with the parsed output of the corresponding fulltext
    for each of the documents in your project.

    Linking will be attempted using your project's link_map.csv.
    See `deet.processors.linker` for more details.

    """
    from deet.processors.linker import DocumentReferenceLinker, LinkingStrategy

    deet_project: DeetProject = typer_context.obj.project
    processed_annotation_data = deet_project.process_data()

    linker = DocumentReferenceLinker(
        references=processed_annotation_data.documents,
        document_base_dir=deet_project.pdf_dir_abspath,
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

    # TODO: Nice message explaining what happened
    console.print(
        info_panel(
            render_template(
                "project/linked",
                linked_documents=linked_documents,
                documents=processed_annotation_data.documents,
            )
        )
    )


@app.command()
def test_llm_config(
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
        load_or_init_config,
    )
    from deet.extractors.llm_data_extractor import LLMDataExtractor

    config = load_or_init_config(config_path)
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
