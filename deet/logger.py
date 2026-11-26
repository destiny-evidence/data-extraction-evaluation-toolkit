"""Loguru setup for the deet CLI."""

from functools import cache

from loguru import logger

from deet.settings import get_settings


@cache
def configure_logging() -> None:
    """Send deet's log records to deet.log in place of loguru's stderr handler."""
    logger.enable("deet")
    logger.remove(0)
    logger.add("deet.log", level=get_settings().log_level, rotation="500 mb")
