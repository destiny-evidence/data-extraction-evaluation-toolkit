import json
from pathlib import Path

import pytest


@pytest.fixture
def valid_parsed_pdf():
    with Path.open("tests/test_files/output/test_file_for_parser.md") as infile:
        return infile.read().lower()


@pytest.fixture
def valid_parsed_epub():
    with Path.open("tests/test_files/output/conrad-epub-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def valid_parsed_html():
    with Path.open("tests/test_files/output/conrad-html-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def sample_eppi_data() -> dict:
    """Load real EPPI data from test file for integration tests."""
    sample_file = Path("tests/test_files/input/sample_eppi.json")
    with sample_file.open() as f:
        return json.load(f)
