from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from deet.exceptions import (
    EmptyPdfExtractionError,
    FileParserMismatchError,
    InvalidFileTypeError,
    InvalidInputFileTypeError,
    InvalidOutputFileTypeError,
)
from deet.processors.parser import (
    DocumentParser,
    InputFileType,
    MarkerParser,
    PandocParser,
    ParsedOutput,
    PdfminerParser,
)
from deet.utils.assess_text_quality import check_language


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
        "deet.processors.parser.converter",
        lambda _: dummy,
    )
    return dummy


@pytest.fixture
def mock_text_from_rendered(monkeypatch):
    """Stub `marker.output.text_from_rendered`."""
    # It returns (text, extension, images)
    monkeypatch.setattr(
        "deet.processors.parser.text_from_rendered",
        lambda _: ("dummy markdown text", "md", []),
    )


@pytest.fixture
def mock_text_from_rendered_img_meta(monkeypatch):
    """Stub `marker.output.text_from_rendered`."""
    dummy_img = Image.new("RGB", (10, 10))
    monkeypatch.setattr(
        "deet.processors.parser.text_from_rendered",
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
        "deet.processors.parser.pypandoc.convert_file",
        lambda file,
        to,
        **kwargs: f"converted {file} to {to} ({kwargs.get('format', '')})",
    )


@pytest.fixture
def mock_check_language(monkeypatch):
    """Stub the language checker."""
    monkeypatch.setattr(
        "deet.processors.parser.check_language",
        lambda txt, lang=None, threshold=0.2: txt.strip() != "not english",  # noqa: ARG005
    )


@pytest.fixture
def mock_pdfminerparser_parse(monkeypatch):
    """Stub PdfminerParser.parse to avoid actual PDF parsing."""

    def _stub_parse(
        cls,
        input_,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        return ParsedOutput(text="dummy pdfminer text")

    monkeypatch.setattr(
        "deet.processors.parser.PdfminerParser.parse",
        classmethod(_stub_parse),
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
            file="badfile.exe", permitted_file_enum_list=MarkerParser.input_types
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


def test_documentparser_parser_none_raises_value_error(
    mock_pypandoc, mock_check_language
):
    """If default parser for file type is None, __call__ should raise ValueError."""
    # create a parser that purposely sets default to None
    p = DocumentParser(parsers=None)

    # monkeypatch detect_filetype to return PDF
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        DocumentParser,
        "detect_filetype",
        lambda *args, **kwargs: "pdf",  # noqa: ARG005
    )  # PT011: give a match string so Ruff knows this is a real test
    with pytest.raises(ValueError, match="no parser supplied."):
        p("any.pdf")
    monkeypatch.undo()


def test_markerparser_success(
    fake_converter, mock_text_from_rendered, mock_check_language
):
    """When Marker is used for a PDF, the returned text matches the stub."""
    parser = DocumentParser()
    parsed_out = parser("any.pdf", parser=MarkerParser)
    assert isinstance(parsed_out, ParsedOutput)
    assert isinstance(parsed_out.text, str)
    assert parsed_out.text == "dummy markdown text"
    assert parsed_out.images is None
    assert parsed_out.metadata is None


def test_markerparser_returns_metadata_and_images(
    fake_converter, mock_text_from_rendered_img_meta, mock_check_language
):
    parser = DocumentParser()
    result = parser(
        "any.pdf", parser=MarkerParser, return_metadata=True, return_images=True
    )
    assert isinstance(result, ParsedOutput)
    assert result.text == "dummy markdown text"
    # Metadata comes from rendered.metadata
    assert result.metadata == {"author": "Nik", "year": 2025}
    assert isinstance(result.images, dict)  # images
    for img in result.images.values():
        assert isinstance(img, Image.Image)


def test_pdfminerparser_success(mock_pdfminerparser_parse, mock_check_language):
    """When PdfminerParser is used for a PDF, the returned text matches the stub."""
    parser = DocumentParser()
    parsed_out = parser("any.pdf", parser=PdfminerParser)
    assert isinstance(parsed_out, ParsedOutput)
    assert isinstance(parsed_out.text, str)
    assert parsed_out.text == "dummy pdfminer text"
    assert parsed_out.images is None
    assert parsed_out.metadata is None


def test_pdfminerparser_raises_on_empty_extraction(tmp_path):
    """PdfminerParser raises EmptyPdfExtractionError when PDF has no text."""
    # Minimal valid PDF (empty page) - PDF Association smallest-possible-pdf-1.0
    minimal_pdf_bytes = (
        b"%PDF-1.0\n"
        b"1 0 obj<< /Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<< /Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<< /Type/Page/Parent 2 0 R/Resources<<>>/MediaBox[0 0 9 9]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n"
        b"0000000101 00000 n \ntrailer<< /Root 1 0 R/Size 4>>\nstartxref\n174\n%%EOF"
    )
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(minimal_pdf_bytes)
    with pytest.raises(EmptyPdfExtractionError, match="no extractable text"):
        PdfminerParser.parse(input_=empty_pdf)


def test_pdfminerparser_raises_on_metadata_or_images(
    mock_pdfminerparser_parse, mock_check_language
):
    """PdfminerParser raises when return_metadata or return_images requested."""
    parser = DocumentParser()
    with pytest.raises(InvalidOutputFileTypeError):
        parser("any.pdf", parser=PdfminerParser, return_metadata=True)
    with pytest.raises(InvalidOutputFileTypeError):
        parser("any.pdf", parser=PdfminerParser, return_images=True)


def test_documentparser_default_pdf_uses_pdfminer(
    mock_pdfminerparser_parse, mock_check_language
):
    """When no parser is supplied for a PDF, the default PdfminerParser is used."""
    parser = DocumentParser()
    parsed_out = parser("doc.pdf")
    assert parsed_out.text == "dummy pdfminer text"


def test_parse_epub_success(mock_pypandoc, mock_check_language):
    parser = DocumentParser()
    parsed_out = parser("book.epub", parser=PandocParser)
    assert isinstance(parsed_out, ParsedOutput)
    assert parsed_out.text == "converted book.epub to md (epub)"
    assert parsed_out.images is None
    assert parsed_out.metadata is None


def test_parse_html_success(mock_pypandoc, mock_check_language):
    parser = DocumentParser()
    parsed_out = parser("page.html", parser=PandocParser)
    assert isinstance(parsed_out, ParsedOutput)
    assert parsed_out.text == "converted page.html to md (html)"
    assert parsed_out.images is None
    assert parsed_out.metadata is None


def test_pandocparser_raises_on_metadata_or_images(mock_pypandoc, mock_check_language):
    parser = DocumentParser()
    with pytest.raises(InvalidOutputFileTypeError):
        parser("book.epub", parser=PandocParser, return_metadata=True)
    with pytest.raises(InvalidOutputFileTypeError):
        parser("book.epub", parser=PandocParser, return_images=True)


def test_documentparser_default_parsers(mock_pypandoc, mock_check_language):
    """When no parser is supplied, the default one for the file type is used."""
    parser = DocumentParser()
    parsed_out = parser("anything.epub")
    # default for epub is PANDOC
    assert parsed_out.text == "converted anything.epub to md (epub)"

    parsed_out2 = parser("page.html")
    assert parsed_out2.text == "converted page.html to md (html)"


def test_documentparser_missing_filetype_raises(tmp_txt_file):
    parser = DocumentParser()
    # force a unsupported file extension
    with pytest.raises(InvalidInputFileTypeError):
        parser(tmp_txt_file)


def test_documentparser_output_file(tmp_path, mock_pypandoc, mock_check_language):
    """When out_path is supplied, parsed text is written & the text is returned."""
    parser = DocumentParser()
    out_path = tmp_path / "out.md"
    result = parser("book.epub", out_path=out_path)
    assert result.text == "converted book.epub to md (epub)"
    assert out_path.read_text() == "converted book.epub to md (epub)"


def test_write_files(tmp_path):
    parser = DocumentParser()
    out = tmp_path / "nested" / "file.md"
    txt = "Hello, world!"
    parser.write_files(
        out_path=out,
        parser=PandocParser,
        write_metadata=False,
        write_images=False,
        text=txt,
    )
    assert out.read_text() == txt


def test_write_files_with_metadata_and_images(tmp_path):
    parser = DocumentParser()
    out = tmp_path / "file.md"
    text = "Hello, world!"
    metadata = {"author": "Nik"}
    images = {"img1.jpg": Image.new("RGB", (10, 10))}
    parser.write_files(
        out_path=out,
        parser=MarkerParser,
        write_metadata=True,
        write_images=True,
        text=text,
        metadata=metadata,
        images=images,
    )
    assert (tmp_path / "file.md").exists()
    assert (tmp_path / "file.json").exists()
    assert any(f.suffix == ".jpg" for f in tmp_path.iterdir())


def test_check_language_quality_en_success():
    proper_english = """
        this is some proper english text. no bad grammar, no
        bad spelling either.
    """
    assert check_language(proper_english, lang="en")


def test_check_language_quality_en_fail():
    bad_english = """hufdshuifhureahuifr."""

    assert not check_language(bad_english, lang="en")


def test_check_language_unimplemented_lang():
    gutes_deutsch = "dies ist ein deutscher satz."
    with pytest.raises(ValueError, match="'de' is not a valid Language"):
        check_language(gutes_deutsch, lang="de")


def test_language_quality_in_pydantic_model():
    parsed_data = ParsedOutput(text="this is an english sentence.")

    assert isinstance(parsed_data, ParsedOutput)
    assert isinstance(parsed_data.text, str)


def test_language_quality_in_pydantic_model_fails():
    with pytest.raises(ValidationError):
        ParsedOutput(text="hufdshuifhureahuifr")


def test_explicit_filetype_for_file(mock_pypandoc, mock_check_language):
    parser = DocumentParser()
    # Explicitly provide filetype for a file
    result = parser(
        "book.epub", parser=PandocParser, input_file_type=InputFileType.EPUB
    )
    assert result.text == "converted book.epub to md (epub)"


def test_parse_jats_xml_file(mock_pypandoc, mock_check_language):
    parser = DocumentParser()
    # simulate parsing a JATS/XML file
    result = parser("article.xml", parser=PandocParser, input_type=InputFileType.XML)
    assert result.text == "converted article.xml to md (jats)"


def test_parse_jats_xml_string(monkeypatch, mock_check_language):
    # simulate xml/jats as str in memory
    monkeypatch.setattr(
        "deet.processors.parser.pypandoc.convert_text",
        lambda text, to, **kwargs: (  # noqa: ARG005
            f"converted string to {to} ({kwargs.get('format', '')})"
        ),
    )
    parser = DocumentParser()
    jats_string = "<article><body>JATS content</body></article>"
    result = parser(
        jats_string,
        parser=PandocParser,
        input_type=InputFileType.XML,
        input_is_string=True,
    )
    assert result.text == "converted string to md (jats)"


def test_parse_jats_xml_string_missing_filetype(monkeypatch, mock_check_language):
    monkeypatch.setattr(
        "deet.processors.parser.pypandoc.convert_text",
        lambda text, to, **kwargs: (  # noqa: ARG005
            f"converted string to {to} ({kwargs.get('format', '')})"
        ),
    )
    parser = DocumentParser()
    jats_string = "<article><body>JATS content</body></article>"
    # Should raise error if input_is_string and input_file_type not provided
    with pytest.raises(InvalidInputFileTypeError):
        parser(
            jats_string,
            parser=PandocParser,
            input_is_string=True,
        )
