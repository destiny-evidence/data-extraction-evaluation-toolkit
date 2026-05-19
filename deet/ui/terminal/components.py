"""Rich UI components for the terminal."""

from rich import box
from rich.align import Align
from rich.console import RenderableType
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


def deprecation_panel(content: RenderableType) -> Panel:
    """Return a panel for deprecation warnings."""
    return Panel(
        content,
        title="[bold red]Deprecation warning[/bold red]",
        border_style="red",
        expand=False,
        padding=(1, 2),
    )


def wizard_header(name: str, current_step: int, total_steps: int) -> Align:
    """Create a header at the top of a wizard detailing progress through fields."""
    header = Panel(
        f"[bold]Setup {name} {current_step}/{total_steps}[/]",
        border_style="bright_blue",
        width=30,
    )
    return Align.center(header)


def wizard_field_help(field: str, help_text: str) -> Align:
    """Print help text to the right."""
    help_card = Panel(
        help_text,
        title=f"About: {field}",
        border_style="cyan",
        padding=(1, 2),
        width=40,
        expand=False,
    )
    return Align.right(help_card)
