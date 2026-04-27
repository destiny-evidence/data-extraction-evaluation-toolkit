"""
Definition of data model to add annotations to pydantic model that can be used
in automatic wizard generation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UI:
    """
    Metadata for automatic UI generation.
    Used with Annotated[Type, UI(help="...")].
    """

    help: str = ""
    label: str | None = None
    placeholder: str | None = None
    hidden: bool = False
    valid: str = "Invalid input"
    instructions: str = ""
