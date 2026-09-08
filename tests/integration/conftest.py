import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def tmp_project_workspace(tmp_path_factory):
    """Create a workspace directory that persists for this module."""
    workspace_dir = tmp_path_factory.mktemp("projects")
    previous_cwd = Path.cwd()
    os.chdir(workspace_dir)
    try:
        yield workspace_dir
    finally:
        os.chdir(previous_cwd)
