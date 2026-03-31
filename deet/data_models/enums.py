"""
A store for plain enums, so they can quickly be imported in the CLI.
We can use these to set argument types and defaults, without needing
large imports, that would slow the CLI down during autocomplete, or when
asking for --help.
"""

from enum import StrEnum, auto


class CustomPromptPopulationMethod(StrEnum):
    """Methods of populating prompts."""

    FILE = auto()
    CLI = auto()
