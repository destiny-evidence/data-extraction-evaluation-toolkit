"""Customisations on the loguru logger."""

from loguru import logger

from deet.settings import get_settings

settings = get_settings()

logger.remove(0)
logger.add("deet.log", level=settings.log_level, rotation="500 mb")
