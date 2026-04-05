"""Help text for CLI commands."""

from deet.ui.terminal.render import flow

APP_HELP = flow("""
    deet (data extraction evaluation toolkit) 🚤

    **quickstart** run `deet`

    Use the deet CLI to extract data from documents with LLMs,
    and evaluate extraction by comparing to human-annotated data.

    To run any of the list of commands below, type `deet *command*`,
    and type `deet *command* --help` to see more information about the command.

    For example, `deet extract-data --help`
    will give you more information about how to use the extract-data command.

    Prefix any command with --verbose to see complete log output.

    "Run `deet --install-completion` to enable your shell to autocomplete deet "
    "commands."
""")
