"""Single import for the rich console, and methods to write to it."""

import re
import textwrap
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from inspect import cleandoc

from jinja2 import Environment, PackageLoader, select_autoescape
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
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
error_console = Console(theme=DEET_THEME, stderr=True)


def render_to_console(message: str, level: LogLevel) -> None:
    """Render message to terminal using Rich."""
    style = level.value.lower()
    panel = Panel(
        message,
        title=f"[{style}]{style.upper()}[/]",
    )
    if level in (LogLevel.ERROR, LogLevel.CRITICAL):
        error_console.print(panel, style=style)
    else:
        console.print(panel, style=style)


@contextmanager
def optional_progress(
    iterable: Iterable, *, show_progress: bool = False, label: str = "Processing"
) -> Generator[Iterable, None, None]:
    """
    Context manager that yields an iterable.
    If show_progress is True, uses a Rich progress bar that handles logs gracefully.
    """
    if not show_progress:
        yield iterable
        return

    # Define the "Columns" of your progress bar for a pro look
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            description=label,
            total=len(list(iterable)) if hasattr(iterable, "__len__") else None,
        )

        # We wrap the iterable to update the progress bar automatically
        def progress_iterable() -> Iterator:
            for item in iterable:
                yield item
                progress.advance(task, 1)

        yield progress_iterable()


def flow(text: str) -> str:
    """
    Clean indentation and collapse single newlines into spaces
    to allow text wrapping in terminal.
    """
    if not text:
        return ""
    # 1. Remove leading indentation
    text = cleandoc(text)
    # 2. Replace single newlines with a space, keep double newlines
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text).strip()


_env = Environment(
    loader=PackageLoader("deet", "ui/terminal/templates"),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(name: str, **context) -> str:
    """Load and render a markdown template."""
    filename = f"{name}.md" if not name.endswith(".md") else name

    tmpl = _env.get_template(filename)
    return textwrap.dedent(tmpl.render(**context)).strip()
