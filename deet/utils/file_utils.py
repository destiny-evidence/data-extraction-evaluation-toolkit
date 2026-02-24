"""File/IO utils."""

from pathlib import Path

import deet


def get_package_root() -> Path:
    """Get the package root from the package's __file__ attribute."""
    return Path(deet.__file__).parent
