"""Registry of deprecated commands with instructions on how to use successors."""

from rich.console import Group
from rich.syntax import Syntax

from deet.ui.terminal import console
from deet.ui.terminal.components import deprecation_panel


def export_config_template_legacy() -> None:
    """Return deprecation warning for old export config command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `export-config-template`"
        " has been deprecated.",
        " \nThe config file is automatically created on setting up a project \n",
        Syntax("deet project init", "bash"),
        " \nTo re-create this file from an existing project, run\n",
        Syntax("deet project regenerate-config-template", "bash"),
    )

    console.print(deprecation_panel(body))


def init_linkage_mapping_file_legacy() -> None:
    """Return deprecation warning for old link map command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `init-linkage-mapping-file`"
        " has been deprecated.",
        " \nThe link map file is automatically created on setting up a project \n",
        Syntax("deet project init", "bash"),
        " \nTo re-create this file from an existing project, run\n",
        Syntax("deet project regenerate-link-map", "bash"),
    )

    console.print(deprecation_panel(body))


def link_documents_fulltexts_legacy() -> None:
    """Return deprecation warning for old export link documents command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `link-documents-fulltexts`"
        " has been deprecated.",
        " \nTo link documents to fulltexts, first create a project \n",
        Syntax("deet project init", "bash"),
        " \nAnd then run, run\n",
        Syntax("deet project link", "bash"),
    )

    console.print(deprecation_panel(body))


def init_prompt_csv_legacy() -> None:
    """Return deprecation warning for old init prompt command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `init-prompt-csv`"
        " has been deprecated.",
        " \nThe prompt csv file is automatically created on setting up a project \n",
        Syntax("deet project init", "bash"),
        " \nTo re-create this file from an existing project, run\n",
        Syntax("deet project regenerate-prompt-csv", "bash"),
    )

    console.print(deprecation_panel(body))


def extract_data_legacy() -> None:
    """Return deprecation warning for old extract data command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `extract-data`"
        " has been deprecated.",
        " \nTo extract data, first define a project with \n",
        Syntax("deet project init", "bash"),
        " \nand follow the instructions to set up your project.",
        "\nOnce your project has been set up, you can extract data with\n",
        Syntax("deet experiments extract", "bash"),
    )

    console.print(deprecation_panel(body))


def test_llm_config_legacy() -> None:
    """Return deprecation warning for old test llm command."""
    body = Group(
        "[bold red]deprecated:[/bold red] the command `test-llm-config`"
        " has been deprecated.",
        " \nTo test your llm config, first create a project: \n",
        Syntax("deet project init", "bash"),
        " \nAnd then run, run\n",
        Syntax("deet project test-llm-config", "bash"),
    )

    console.print(deprecation_panel(body))
