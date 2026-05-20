import pytest

from deet.processors.directory_processor import create_documents_from_directory


@pytest.fixture
def mock_pdf_dir(tmp_path, mock_pdfminerparser_parse, mock_check_language):
    """Create a directory of pdfs and md files."""
    pdf_dir = tmp_path / "pdf_dir"
    pdf_dir.mkdir()

    # Files with the same stem (we should only parse md)
    (pdf_dir / "doc_alpha.md").write_text("# Alpha Markdown Content")
    (pdf_dir / "doc_alpha.pdf").write_text("# Alpha PDF content")

    # Files with unique stems
    (pdf_dir / "doc_beta.pdf").write_text("# Beta pdf content")
    (pdf_dir / "doc_gamma.md").write_text("# Gamma Markdown Content")

    # Non-parsed file type that should be skipped completely
    (pdf_dir / "notes.txt").write_text("This text file should be ignored")

    return pdf_dir


def test_create_documents_from_directory(mock_pdf_dir):
    documents = create_documents_from_directory(mock_pdf_dir)

    assert len(documents) == 3
    doc_names = {doc.name for doc in documents}
    assert "doc_alpha.md" in doc_names
    assert "doc_alpha.pdf" not in doc_names
    assert "doc_beta.pdf" in doc_names
    assert "doc_gamma.md" in doc_names
    assert "notes.txt" not in doc_names
