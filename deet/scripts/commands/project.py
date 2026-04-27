# ruff: noqa: PLC0415, B008
"""Sub-commands for project initialisation and configuration."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from deet.data_models.project import DeetProject


import typer
from InquirerPy import inquirer
from pydantic import ValidationError

from deet.processors.converter_register import SupportedImportFormat
from deet.scripts.typer_context import project_required
from deet.settings import DataExtractionSettings, LogLevel
from deet.ui import fail_with_message, notify
from deet.ui.terminal import (
    console,
    continue_after_key,
    render_template,
    run_model_wizard,
)
from deet.ui.terminal.components import info_panel

app = typer.Typer(help="Commands to create and configure deet projects.")


@app.command()
def init(  # noqa: PLR0913
    typer_context: typer.Context,
    *,
    name: str = typer.Option(
        None, "--name", "-n", help="Project name (min 2 characters)"
    ),
    data_path: Path = typer.Option(
        None,
        "--data",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to your gold standard annotation data",
    ),
    data_type: SupportedImportFormat = typer.Option(
        SupportedImportFormat.EPPI_JSON,
        "--format",
        "-t",
        help="Format of your gold standard annotated data.",
    ),
    pdf_dir: Path | None = typer.Option(
        None,
        "--pdfs",
        "-p",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="The folder where your pdfs for data extraction are stored.",
    ),
    force_overwrite: bool = typer.Option(
        False,  # noqa: FBT003
        "--force",
        "-f",
        help="Overwrite existing project data",
    ),
) -> None:
    """
    Initialise a new project.

    Leave command line arguments empty to enter interactive wizard.
    """
    from deet.data_models.project import DeetProject

    existing_project: DeetProject = typer_context.obj.project

    if any([name, data_path, pdf_dir]):
        if existing_project and not force_overwrite:
            fail_with_message("Project already exists. ")
        try:
            project = DeetProject(
                name=name,
                gold_standard_data_path=data_path,
                gold_standard_data_format=data_type,
                pdf_dir=pdf_dir,
            )
        except ValidationError as e:
            fail_with_message(f"Invalid project configuration:\n{e}")
        project.setup()
        return

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
    settings.dump_to_env()

    console.clear()
    console.print(info_panel(render_template("project/success.md", project=project)))


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
        file_path=deet_project.link_map_path
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
    typer_context: typer.Context,
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
        load_config_from_typer_context,
    )
    from deet.extractors.llm_data_extractor import LLMDataExtractor

    config = load_config_from_typer_context(typer_context, config_path)
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
