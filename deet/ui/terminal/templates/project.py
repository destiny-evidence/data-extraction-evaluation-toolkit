"""UI text for project setup."""

from inspect import cleandoc

from deet.data_models.project import DeetProject


def project_init_md() -> str:
    """Markdown text to render at start of project creation."""
    return cleandoc(
        """
        ## Welcome to `deet`
        Let's start configuring a data extraction project.

        The following set-up wizard will collect information required to configure
        your data extraction project. Help text for each question is shown on the
        right hand side of your screen. Note that some questions will only accept
        valid responses. If you try to enter an invalid response, a message will
        describe validation rules at the bottom of your screen.

        You can press Ctrl-c at any point to quit.
        """
    )


def configure_env_md() -> str:
    """Markdown text to introduce user to .env wizard."""
    return cleandoc(
        """
        ## Let's store your API keys

        Since you have chosen to store credentials specific to this project,
        let's collect these and save them to a .env file in this directory.
        Make sure you do not commit or otherwise share this file publicly.
        """
    )


def project_sucess_md(project: DeetProject) -> str:
    """Markdown on successful creation of project."""
    return cleandoc(
        f"""
        ## Success! Project {project.name} is ready to go
        Project root: `{project.root}`


        ### Next steps

        1. **Define your prompts:** open your prompt definition file,
        and add a prompt in the prompt column for each attribute you want
        to extract. `{project.prompt_csv_path}`

        2. **Point your documents to files:** open your link map and add a
        filename for the full text you want to link each document to
        `{project.link_map_path}`

        3. **Link documents:** link documents to full texts using the link map. Run
        ```sh
        deet project link
        ```

        4. **Run a data extraction experiment:** Use the wizard to set up a
        data extraction experiment using your data. Run
        ```sh
        deet run extract
        ```
        If you prefer to skip the wizard, edit the config file `{project.config_path}`
        and pass this as a command line argument
        ```sh
        deet run --config-path {project.config_path} extract
        ```
        """
    )
