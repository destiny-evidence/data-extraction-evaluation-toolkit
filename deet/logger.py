"""Customisations on the loguru logger."""

import sys

from loguru import logger

from deet.settings import get_settings

settings = get_settings()

logger.remove(0)
logger.add("deet.log", level=settings.log_level, rotation="500 mb")
logger.add(sys.stderr, level=settings.log_level)
