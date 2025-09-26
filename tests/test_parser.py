from pathlib import Path

import pytest
from PIL import Image

from app.exceptions import (
    InvalidFileTypeError,
    InvalidInputFileTypeError,
    InvalidOutputFileTypeError,
)
from app.parser import DocumentParser, InputFileType, MarkerParser, PandocParser


@pytest.fixture
def fake_converter(monkeypatch):
    """
    Stub the global `converter` that comes from `marker.converters.pdf`.
    It should return an object that `text_from_rendered` can consume.
    """

    # The real converter returns a dict; we only need the text part.
    class DummyRendered:
        metadata = {"author": "Nik", "year": 2025}

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
    # It returns (text, extension, images)
    monkeypatch.setattr(
        "app.parser.text_from_rendered",
        lambda _: ("dummy markdown text", "md", []),
    )


@pytest.fixture
def mock_text_from_rendered_img_meta(monkeypatch):
    """Stub `marker.output.text_from_rendered`."""
    dummy_img = Image.new("RGB", (10, 10))
    monkeypatch.setattr(
        "app.parser.text_from_rendered",
        lambda _: (
            "dummy markdown text",
            "md",
            {"img1.jpg": dummy_img, "img2.jpg": dummy_img},
        ),
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


def test_detect_filetype_invalid_input_output():
    with pytest.raises(InvalidFileTypeError) as exc:
        DocumentParser.detect_filetype("badfile.exe")
    assert "not permitted" in str(exc.value)


def test_detect_filetype_invalid_input():
    with pytest.raises(InvalidInputFileTypeError) as exc:
        DocumentParser.detect_filetype(
            file="badfile.exe", permitted_file_enum_list=MarkerParser.input_file_types
        )
    assert "not permitted" in str(exc.value)


def test_detect_filetype_invalid_output():
    with pytest.raises(InvalidOutputFileTypeError) as exc:
        DocumentParser.detect_filetype("badfile.exe", MarkerParser.output_file_types)
    assert "not permitted" in str(exc.value)


def test_documentparser_unknown_parser():
    """If the parser argument is not a ParserLibrary member, it should raise."""
    parser = DocumentParser()
    with pytest.raises(FileParserMismatchError):
        # passing a string that is not a ParserLibrary
        parser("book.epub", parser="unknown")


# def test_documentparser_parser_none_raises_value_error(mock_pypandoc, mock_is_english):
#     """If the default parser for a file type is None, __call__ should raise ValueError."""
#     # create a parser that purposely sets default to None
#     p = DocumentParser(default_parser_pdf=None)

#     # monkeypatch detect_filetype to return PDF
#     monkeypatch = pytest.MonkeyPatch()
#     monkeypatch.setattr(
#         DocumentParser,
#         "detect_filetype",
#         lambda self,  # noqa: ARG005
#         input_file: InputFileType.PDF,  # Ensure self is included, # noqa: ARG005
#     )

#     # PT011: give a match string so Ruff knows this is a real test
#     with pytest.raises(ValueError, match="no parser supplied."):
#         p("any.pdf")
#     monkeypatch.undo()


# def test_markerparser_success(fake_converter, mock_text_from_rendered, mock_is_english):
#     """When Marker is used for a PDF, the returned text matches the stub."""
#     parser = DocumentParser()
#     txt = parser("any.pdf", parser=MarkerParser)
#     assert isinstance(txt, tuple)
#     assert txt[0] == "dummy markdown text"
#     assert len(txt) == 1


# def test_markerparser_returns_metadata_and_images(
#     fake_converter, mock_text_from_rendered_img_meta, mock_is_english
# ):
#     parser = DocumentParser()
#     result = parser(
#         "any.pdf", parser=MarkerParser, return_metadata=True, return_images=True
#     )
#     assert isinstance(result, tuple)
#     assert result[0] == "dummy markdown text"
#     # Metadata comes from rendered.metadata
#     assert result[1] == {"author": "Nik", "year": 2025}
#     assert isinstance(result[2], dict)  # images
#     for img in result[2].values():
#         assert isinstance(img, Image.Image)


# def test_parse_epub_success(mock_pypandoc, mock_is_english):
#     parser = DocumentParser()
#     txt = parser("book.epub", parser=PandocParser)
#     assert isinstance(txt, str)
#     assert txt == "converted book.epub to md (epub)"


# def test_parse_html_success(mock_pypandoc, mock_is_english):
#     parser = DocumentParser()
#     txt = parser("page.html", parser=PandocParser)
#     assert txt == "converted page.html to md (html)"


# def test_pandocparser_raises_on_metadata_or_images(mock_pypandoc, mock_is_english):
#     parser = DocumentParser()
#     with pytest.raises(InvalidOutputFileTypeError):
#         parser("book.epub", parser=PandocParser, return_metadata=True)
#     with pytest.raises(InvalidOutputFileTypeError):
#         parser("book.epub", parser=PandocParser, return_images=True)


# def test_documentparser_default_parsers(mock_pypandoc, mock_is_english):
#     """When no parser is supplied, the default one for the file type is used."""
#     parser = DocumentParser()
#     txt = parser("anything.epub")
#     # default for epub is PANDOC
#     assert txt == "converted anything.epub to md (epub)"

#     txt2 = parser("page.html")
#     assert txt2 == "converted page.html to md (html)"


# def test_documentparser_missing_filetype_raises(tmp_txt_file):
#     parser = DocumentParser()
#     # force a unsupported file extension
#     with pytest.raises(InvalidInputFileTypeError):
#         parser(tmp_txt_file)


# def test_documentparser_output_file(tmp_path, mock_pypandoc, mock_is_english):
#     """When an out_path is supplied, the parsed text is written and the text is returned."""
#     parser = DocumentParser()
#     out = tmp_path / "out.md"
#     result = parser("book.epub", out_path=out)
#     assert result == "converted book.epub to md (epub)"
#     assert out.read_text() == "converted book.epub to md (epub)"


# def test_write_files(tmp_path):
#     parser = DocumentParser()
#     out = tmp_path / "nested" / "file.md"
#     txt = "Hello, world!"
#     parser.write_files(
#         out_path=out,
#         parser=PandocParser,
#         write_metadata=False,
#         write_images=False,
#         text=txt,
#     )
#     assert out.read_text() == txt


# def test_write_files_with_metadata_and_images(tmp_path):
#     parser = DocumentParser()
#     out = tmp_path / "file.md"
#     text = "Hello, world!"
#     metadata = {"author": "Nik"}
#     images = {"img1.jpg": Image.new("RGB", (10, 10))}
#     parser.write_files(
#         out_path=out,
#         parser=MarkerParser,
#         write_metadata=True,
#         write_images=True,
#         text=text,
#         metadata=metadata,
#         images=images,
#     )
#     assert (tmp_path / "file.md").exists()
#     assert (tmp_path / "file.json").exists()
#     assert any(f.suffix == ".jpg" for f in tmp_path.iterdir())


def test_check_text_is_english_success():
    proper_english = """
        this is some proper english text. no bad grammar, no
        bad spelling either.
    """
    assert DocumentParser.check_text_is_english(proper_english)


def test_check_text_is_english_fail():
    bad_english = """hufdshuifhureahuifr."""

    assert not DocumentParser.check_text_is_english(bad_english)
