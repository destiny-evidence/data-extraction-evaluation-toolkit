from pathlib import Path

import pytest

from app.parser import (
    DocumentParser,
    FileParserMismatchError,
    InputFileType,
    InvalidInputFileTypeError,
    ParserLibrary,
)


@pytest.fixture
def fake_converter(monkeypatch):
    """
    Stub the global `converter` that comes from `marker.converters.pdf`.
    It should return an object that `text_from_rendered` can consume.
    """

    # The real converter returns a dict; we only need the text part.
    class DummyRendered:
        pass

    dummy = DummyRendered()
    # `converter(file)` simply returns the dummy object
    monkeypatch.setattr(
        "app.parser.converter",
        lambda _: dummy,
    )
    return dummy


@pytest.fixture
def mock_text_from_rendered(monkeypatch):
    """Stub `marker.output.text_from_rendered`."""
    # It returns (text, metadata, images)
    monkeypatch.setattr(
        "app.parser.text_from_rendered",
        lambda _: ("dummy markdown text", {}, []),
    )


@pytest.fixture
def mock_pypandoc(monkeypatch):
    """Stub `pypandoc.convert_file`."""
    monkeypatch.setattr(
        "app.parser.pypandoc.convert_file",
        lambda file, to, format: f"converted {file} to {to} ({format})",
    )


@pytest.fixture
def mock_is_english(monkeypatch):
    """Stub the simple English checker."""
    monkeypatch.setattr(
        "app.parser.is_english",
        lambda txt: txt.strip() != "not english",
    )


@pytest.fixture
def tmp_txt_file(tmp_path):
    """Create a temporary text file that can be used as a dummy input."""
    p = tmp_path / "sample.txt"
    p.write_text("some content")
    return p


def test_detect_filetype_valid():
    assert DocumentParser.detect_filetype("foo.pdf") == InputFileType.PDF
    assert DocumentParser.detect_filetype(Path("bar.epub")) == InputFileType.EPUB
    assert DocumentParser.detect_filetype("/tmp/test.html") == InputFileType.HTML  # noqa: S108


def test_detect_filetype_invalid():
    with pytest.raises(InvalidInputFileTypeError) as exc:
        DocumentParser.detect_filetype("badfile.exe")
    assert "not permitted" in str(exc.value)


def test_documentparser_unknown_parser():
    """If the parser argument is not a ParserLibrary member, it should raise."""
    parser = DocumentParser()
    with pytest.raises(FileParserMismatchError):
        # passing a string that is not a ParserLibrary
        parser("book.epub", parser="unknown")


def test_documentparser_parser_none_raises_value_error(mock_pypandoc, mock_is_english):
    """If the default parser for a file type is None, __call__ should raise ValueError."""
    # create a parser that purposely sets default to None
    p = DocumentParser(default_parser_pdf=None)

    # monkeypatch detect_filetype to return PDF
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        DocumentParser,
        "detect_filetype",
        lambda self,  # noqa: ARG005
        input_file: InputFileType.PDF,  # Ensure self is included, # noqa: ARG005
    )

    # PT011: give a match string so Ruff knows this is a real test
    with pytest.raises(ValueError, match="no parser supplied."):
        p("any.pdf")
    monkeypatch.undo()


def test_parse_pdf_marker_success(
    fake_converter, mock_text_from_rendered, mock_is_english
):
    """When Marker is used for a PDF, the returned text matches the stub."""
    txt = DocumentParser.parse_pdf("any.pdf", ParserLibrary.MARKER)
    assert txt == "dummy markdown text"


def test_parse_pdf_bad_parser():
    """Using an unsupported parser for PDF raises FileParserMismatchError."""
    with pytest.raises(FileParserMismatchError):
        DocumentParser.parse_pdf("any.pdf", ParserLibrary.PANDOC)


def test_parse_epub_success(mock_pypandoc, mock_is_english):
    txt = DocumentParser.parse_epub("book.epub", ParserLibrary.PANDOC)
    assert txt == "converted book.epub to md (epub)"


def test_parse_html_success(mock_pypandoc, mock_is_english):
    txt = DocumentParser.parse_html("page.html", ParserLibrary.PANDOC)
    assert txt == "converted page.html to md (html)"


def test_parse_pdf_bad_parser_file_combo():
    with pytest.raises(FileParserMismatchError):
        DocumentParser.parse_pdf(file="test.pdf", parser=ParserLibrary.PANDOC)


def test_parse_epub_bad_parser_file_combo():
    with pytest.raises(FileParserMismatchError):
        DocumentParser.parse_epub(file="test.epub", parser=ParserLibrary.MARKER)


def test_parse_html_bad_parser_file_combo():
    with pytest.raises(FileParserMismatchError):
        DocumentParser.parse_html(file="test.html", parser=ParserLibrary.MARKER)


def test_documentparser_default_parsers(mock_pypandoc, mock_is_english):
    """When no parser is supplied, the default one for the file type is used."""
    parser = DocumentParser()
    txt = parser("anything.epub")
    # default for epub is PANDOC
    assert txt == "converted anything.epub to md (epub)"

    txt2 = parser("page.html")
    assert txt2 == "converted page.html to md (html)"


def test_documentparser_missing_filetype_raises(tmp_txt_file):
    parser = DocumentParser()
    # force a unsupported file extension
    with pytest.raises(InvalidInputFileTypeError):
        parser(tmp_txt_file)


def test_documentparser_output_file(tmp_path, mock_pypandoc, mock_is_english):
    """When an output_file is supplied, the parsed text is written and the path is returned."""
    parser = DocumentParser()
    out = tmp_path / "out.md"
    result = parser("book.epub", output_file=out)
    assert result == str(out)
    assert out.read_text() == "converted book.epub to md (epub)"


def test_write_to_file(tmp_path):
    out = tmp_path / "nested" / "file.md"
    txt = "Hello, world!"
    res = DocumentParser.write_to_file(txt, out)
    assert res == str(out)
    assert out.read_text() == txt


def test_check_text_is_english_success():
    proper_english = """
        this is some proper english text. no bad grammar, no
        bad spelling either.
    """
    assert DocumentParser.check_text_is_english(proper_english)


def test_check_text_is_english_fail():
    bad_english = """hufdshuifhureahuifr."""

    assert not DocumentParser.check_text_is_english(bad_english)
