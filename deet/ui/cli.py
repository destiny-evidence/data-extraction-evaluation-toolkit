"""CLI interface to run fasthtml app."""

import uvicorn


def main() -> None:
    """Run the app defined in web with uvicorn."""
    # Uvicorn imports app by module string
    uvicorn.run("deet.ui.web:app", host="127.0.0.1", port=8000)
