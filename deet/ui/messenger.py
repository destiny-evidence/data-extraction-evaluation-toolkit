"""Interface to pass messages to UI(s)."""

from deet.logger import logger
from deet.settings import LogLevel
from deet.ui.terminal import render_to_console


def notify(message: str, level: LogLevel = LogLevel.INFO) -> None:
    """Send messages to logger and to UIs (currently the console)."""
    logger.bind(is_echo=True).log(level.value, message)
    render_to_console(message, level)
