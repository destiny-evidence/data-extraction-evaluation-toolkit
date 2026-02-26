"""Utilities to help with the CLI."""

from pathlib import Path
from uuid import UUID


def get_last_pipeline_run(pipeline_dir: Path) -> tuple[UUID, Path]:
    """Get the last pipeline run from a folder of pipeline runs."""
    valid_dirs = []
    for sub in pipeline_dir.iterdir():
        try:
            UUID(sub.name)
            valid_dirs.append(sub)
        except ValueError:
            pass

    if len(valid_dirs) == 0:
        no_runs = f"No valid run directories in {dir}"
        raise ValueError(no_runs)

    valid_dirs.sort(key=lambda p: p.name)
    last_path = valid_dirs[-1]
    last_uuid = UUID(last_path.name)
    return last_uuid, last_path
