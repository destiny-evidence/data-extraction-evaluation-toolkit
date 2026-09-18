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


class EvaluationStrategyName(StrEnum):
    """
    A list of allowable names for evaluation strategies.

    Note: make sure that each of these is in the strategy registry in
    deet.data_models.evaluation_strategies.__init__.py.
    """

    NONE = auto()
    DEV_VAL_TEST = auto()
