"""Customisations on the loguru logger."""

import sys

from loguru import logger

logger.remove(0)
logger.add("app.log", level="DEBUG", rotation="500 mb")
logger.add(sys.stderr, level="DEBUG")
