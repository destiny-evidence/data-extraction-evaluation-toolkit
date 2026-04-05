"""Rich UI components for the terminal."""

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel


def info_panel(content: str, title: str = "INFO") -> Panel:
    """Return a styled box for displaying Markdown content."""
    return Panel(
        Markdown(content),
        title=title,
        border_style="bright_blue",
        box=box.DOUBLE,
        padding=(1, 2),
        expand=False,
        width=120,
    )
