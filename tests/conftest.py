from pathlib import Path

import pytest


@pytest.fixture
def valid_parsed_pdf():
    with Path.open(
        "tests/test_files/output/fd7d92bb-a0f9-4b52-8fa6-a5a52ca9c0ee.md"
    ) as infile:
        return infile.read().lower()


@pytest.fixture
def valid_parsed_epub():
    with Path.open("tests/test_files/output/conrad-epub-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def valid_parsed_html():
    with Path.open("tests/test_files/output/conrad-html-parsed.md") as infile:
        return infile.read()
