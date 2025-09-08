from pathlib import Path

import pytest


@pytest.fixture
def valid_parsed_pdf():
    with Path.open("tests/test_files/output/vieira_2008.md") as infile:
        return infile.read().lower()


@pytest.fixture
def valid_parsed_epub():
    with Path.open("tests/test_files/output/conrad-epub-parsed.md") as infile:
        return infile.read()


@pytest.fixture
def valid_parsed_html():
    with Path.open("tests/test_files/output/conrad-html-parsed.md") as infile:
        return infile.read()
