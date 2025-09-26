"""NOTE: these take eons to run, and currently don't work for pdf, as the old .md file doesn't match the new one."""

import re
from pathlib import Path

import pytest

from app.parser import (
    DocumentParser,
    FileParserMismatchError,
    InvalidInputFileTypeError,
    ParserLibrary,
)


def _normalise(text: str) -> str:
    """Normalise text used in the tests."""
    text = text.strip()
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.lower()
    return re.sub(r"(?m)^\s*(#\s*)", r"\1", text)


def test_real_pdf_parses_correctly(valid_parsed_pdf):
    parser = DocumentParser()
    parsed = parser(
        input_file="tests/test_files/input/fd7d92bb-a0f9-4b52-8fa6-a5a52ca9c0ee.pdf"
    ).lower()
    assert _normalise(parsed) == _normalise(valid_parsed_pdf)


def test_real_epub_parses_correctly(valid_parsed_epub):
    parser = DocumentParser()
    parsed = parser("tests/test_files/input/conrad.epub")
    assert parsed.strip() == valid_parsed_epub.strip()


def test_real_html_parses_correctly(valid_parsed_html):
    parser = DocumentParser()
    parsed = parser("tests/test_files/input/conrad.html")
    assert parsed.strip() == valid_parsed_html.strip()


def test_output_file_is_written(tmp_path, valid_parsed_pdf):
    parser = DocumentParser()
    out = tmp_path / "out.md"
    parser(
        "tests/test_files/input/fd7d92bb-a0f9-4b52-8fa6-a5a52ca9c0ee.pdf",
        output_file=out,
    )
    assert Path(out).is_file()


def test_invalid_filetype_raises(tmp_path):
    bad_file = tmp_path / "oops.exe"
    bad_file.write_text("...")

    parser = DocumentParser()
    with pytest.raises(InvalidInputFileTypeError):
        parser(bad_file)


# def test_parser_override_with_marker_for_pdf(tmp_path):
#     """Explicitly pass the Marker parser for a PDF."""
#     parser = DocumentParser(
#         default_parser_pdf=ParserLibrary.PANDOC
#     )  # force wrong default
#     out_file = tmp_path / "marker.md"
#     result = parser(
#         "tests/test_files/input/Abroms_2008.pdf",
#         parser=ParserLibrary.MARKER,
#         output_file=out_file,
#     )
#     assert result == str(out_file)
#     # The content should match the fixture (Marker produces the same output as the real parser)
#     with Path.open("tests/test_files/output/Abroms_2008.md") as ref:
#         assert out_file.read_text() == ref.read()


def test_bad_parser_for_pdf_raises():
    parser = DocumentParser()
    with pytest.raises(FileParserMismatchError):
        parser("tests/test_files/input/Abroms_2008.pdf", parser=ParserLibrary.PANDOC)
