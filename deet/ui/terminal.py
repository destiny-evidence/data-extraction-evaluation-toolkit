"""Single import for the rich console, and methods to write to it."""

from rich.console import Console
from rich.theme import Theme

from deet.settings import LogLevel

DEET_THEME = Theme(
    {
        "info": "white",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
    }
)

console = Console(theme=DEET_THEME)


def render_to_console(message: str, level: LogLevel) -> None:
    """Render message to terminal using Rich."""
    style = level.value.lower()
    console.print(message, style=style)
