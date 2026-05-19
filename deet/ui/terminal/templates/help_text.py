"""Help text for CLI commands."""

from deet.ui.terminal.render import flow

APP_HELP = flow("""
    deet (data extraction evaluation toolkit) 🚤

    **quickstart** run `deet`

    Use the deet CLI to extract data from documents with LLMs,
    and evaluate extraction by comparing to human-annotated data.

    run `deet project --help` for information on commands to create and configure
    a project.

    run `deet run --help` for information on commands to extract data

    Prefix any command with --verbose to see complete log output.

    Run `deet --install-completion` to enable your shell to autocomplete deet
    commands.
""")
