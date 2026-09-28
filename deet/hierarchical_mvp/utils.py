from __future__ import annotations

import csv
import os
from pathlib import Path

# Import anyio eagerly before DSPy so the real anyio module is registered in
# sys.modules before dspy's LazyImport machinery can install a placeholder.
# FastAPI (pulled in via destiny_sdk.auth) imports anyio.abc and anyio low-level
# modules during import, and an uninitialized LazyImport placeholder causes the
# circular-import failures seen in notebook/Colab environments.
import anyio  # noqa: F401
import anyio.abc  # noqa: F401
import anyio.lowlevel  # noqa: F401

import dspy
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from deet.logger import logger
