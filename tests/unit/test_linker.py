"""Tests for stuff in the processors/linker.py module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from destiny_sdk.references import ReferenceFileInput

from deet.data_models.documents import (
    ContextType,
    Document,
    DocumentIdentity,
    DocumentIDSource,
)
from deet.exceptions import JsonStyleError
from deet.processors.linker import (
    DocumentReferenceLinker,
    DocumentReferenceMapping,
    LinkedInterimPayload,
    LinkingStrategy,
    MappingImporter,
)
from deet.processors.parser import ParsedOutput

# NOTE - didn't really use a realistic ReferenceFileInput (citation) in these tests.
#        that could potentially be remedied by adding a strong example into conftest,
#        and subbing that in to these tests.


# DocumentReferenceMapping related, not integrated elsewhere in the module
def test_document_reference_mapping_valid_pdf(tmp_path):
    """Test creating DocumentReferenceMapping with valid PDF file."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf content")

    mapping = DocumentReferenceMapping(
        document_id=12345678,
        file_path=pdf_file,
    )
    assert mapping.document_id == 12345678
    assert mapping.file_path == pdf_file
    assert mapping.format == "pdf"


def test_document_reference_mapping_valid_md(tmp_path):
    """Test creating DocumentReferenceMapping with valid markdown file."""
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test markdown")

    mapping = DocumentReferenceMapping(
        document_id=87654321,
        file_path=md_file,
    )
    assert mapping.document_id == 87654321
    assert mapping.file_path == md_file
    assert mapping.format == "md"


def test_document_reference_mapping_invalid_document_id_too_short():
    """Test DocumentReferenceMapping raises error for short document_id."""
    with pytest.raises(ValueError, match="8 digits"):
        DocumentReferenceMapping(
            document_id=1234,
            file_path=Path("/fake/path.pdf"),
        )


def test_document_reference_mapping_invalid_document_id_too_long():
    """Test DocumentReferenceMapping raises error for long document_id."""
    with pytest.raises(ValueError, match="8 digits"):
        DocumentReferenceMapping(
            document_id=123456789,
            file_path=Path("/fake/path.pdf"),
        )


def test_document_reference_mapping_file_does_not_exist():
    """Test DocumentReferenceMapping raises error for non-existent file."""
    with pytest.raises(ValueError, match="not a file"):
        DocumentReferenceMapping(
            document_id=12345678,
            file_path=Path("/nonexistent/file.pdf"),
        )


def test_document_reference_mapping_unsupported_format(tmp_path):
    """Test DocumentReferenceMapping raises error for unsupported file format."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("text content")

    with pytest.raises(ValueError, match="not supported"):
        DocumentReferenceMapping(
            document_id=12345678,
            file_path=txt_file,
        )


def test_document_reference_mapping_directory_not_file(tmp_path):
    """Test DocumentReferenceMapping raises error when path is a directory."""
    with pytest.raises(ValueError, match="not a file"):
        DocumentReferenceMapping(
            document_id=12345678,
            file_path=tmp_path,
        )


# LinkedInterimPayload
def test_linked_interim_payload_creation(tmp_path):
    """Test creating LinkedInterimPayload with valid data."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(name="Test Doc", citation=ReferenceFileInput())

    payload = LinkedInterimPayload(
        document_id=12345678,
        file_path=pdf_file,
        format="pdf",
        unlinked_document=doc,
    )
    assert payload.document_id == 12345678
    assert payload.file_path == pdf_file
    assert payload.format == "pdf"
    assert payload.unlinked_document == doc


def test_linked_interim_payload_inherits_validation(tmp_path):
    """Test LinkedInterimPayload inherits DocumentReferenceMapping validation."""
    citation = ReferenceFileInput()
    doc = Document(name="Test Doc", citation=citation)

    with pytest.raises(ValueError, match="8 digits"):
        LinkedInterimPayload(
            document_id=123,
            file_path=Path("/fake/path.pdf"),
            unlinked_document=doc,
        )


# MappingImporter
def test_mapping_importer_init_json(tmp_path):
    """Test MappingImporter initialization with JSON file."""
    json_file = tmp_path / "mapping.json"
    json_file.write_text("{}")

    importer = MappingImporter(mapping_file_path=json_file)
    assert importer.mapping_file_path == json_file
    assert importer.mapping_file_type == "json"


def test_mapping_importer_init_csv(tmp_path):
    """Test MappingImporter initialization with CSV file."""
    csv_file = tmp_path / "mapping.csv"
    csv_file.write_text("document_id,file_path")

    importer = MappingImporter(mapping_file_path=csv_file)
    assert importer.mapping_file_path == csv_file
    assert importer.mapping_file_type == "csv"


def test_mapping_importer_init_invalid_format(tmp_path):
    """Test MappingImporter raises error for invalid file format."""
    txt_file = tmp_path / "mapping.txt"
    txt_file.write_text("content")

    with pytest.raises(ValueError, match="csv or json"):
        MappingImporter(mapping_file_path=txt_file)


def test_mapping_importer_init_with_document_base_dir(tmp_path):
    """Test MappingImporter with valid document_base_dir."""
    json_file = tmp_path / "mapping.json"
    json_file.write_text("{}")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    importer = MappingImporter(
        mapping_file_path=json_file,
        document_base_dir=docs_dir,
    )
    assert importer.document_base_dir == docs_dir


def test_mapping_importer_init_invalid_document_base_dir(tmp_path):
    """Test MappingImporter raises error for invalid document_base_dir."""
    json_file = tmp_path / "mapping.json"
    json_file.write_text("{}")

    with pytest.raises(ValueError, match="must be a dir"):
        MappingImporter(
            mapping_file_path=json_file,
            document_base_dir=tmp_path / "nonexistent",
        )


def test_mapping_importer_load_json_dict_style(tmp_path):
    """Test loading JSON mapping in dict style."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": str(pdf_file),
            }
        )
    )

    importer = MappingImporter(mapping_file_path=json_file)
    mappings = importer.import_mapping()

    assert len(mappings) == 1
    assert mappings[0].document_id == 12345678
    assert mappings[0].file_path == pdf_file


def test_mapping_importer_load_json_array_style(tmp_path):
    """Test loading JSON mapping in array style."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            [
                {"document_id": 12345678, "file_path": str(pdf_file)},
            ]
        )
    )

    importer = MappingImporter(mapping_file_path=json_file)
    mappings = importer.import_mapping()

    assert len(mappings) == 1
    assert mappings[0].document_id == 12345678


def test_mapping_importer_load_json_multiple_entries(tmp_path):
    """Test loading JSON mapping with multiple entries."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf1 = docs_dir / "test1.pdf"
    pdf1.write_text("pdf 1")
    pdf2 = docs_dir / "test2.md"
    pdf2.write_text("# markdown")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": str(pdf1),
                "87654321": str(pdf2),
            }
        )
    )

    importer = MappingImporter(mapping_file_path=json_file)
    mappings = importer.import_mapping()

    assert len(mappings) == 2
    doc_ids = {m.document_id for m in mappings}
    assert doc_ids == {12345678, 87654321}


def test_mapping_importer_load_json_invalid_style(tmp_path):
    """Test loading JSON with invalid style raises JsonStyleError."""
    json_file = tmp_path / "mapping.json"
    json_file.write_text('"just a string"')

    importer = MappingImporter(mapping_file_path=json_file)
    with pytest.raises(JsonStyleError, match="list.*dict"):
        importer.import_mapping()


def test_mapping_importer_load_csv(tmp_path):
    """Test loading CSV mapping."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    csv_file = tmp_path / "mapping.csv"
    csv_file.write_text(f"document_id,file_path\n12345678,{pdf_file}\n")

    importer = MappingImporter(mapping_file_path=csv_file)
    mappings = importer.import_mapping()

    assert len(mappings) == 1
    assert mappings[0].document_id == 12345678


def test_mapping_importer_load_csv_multiple_entries(tmp_path):
    """Test loading CSV mapping with multiple entries."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf1 = docs_dir / "test1.pdf"
    pdf1.write_text("pdf 1")
    pdf2 = docs_dir / "test2.pdf"
    pdf2.write_text("pdf 2")

    csv_file = tmp_path / "mapping.csv"
    csv_file.write_text(f"document_id,file_path\n12345678,{pdf1}\n87654321,{pdf2}\n")

    importer = MappingImporter(mapping_file_path=csv_file)
    mappings = importer.import_mapping()

    assert len(mappings) == 2


def test_mapping_importer_load_csv_missing_columns(tmp_path):
    """Test loading CSV with missing required columns raises error."""
    csv_file = tmp_path / "mapping.csv"
    csv_file.write_text("id,path\n12345678,/fake/path.pdf\n")

    importer = MappingImporter(mapping_file_path=csv_file)
    with pytest.raises(ValueError, match="document_id.*file_path"):
        importer.import_mapping()


def test_mapping_importer_resolve_relative_path_with_base_dir(tmp_path):
    """Test resolving relative paths with document_base_dir."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": "test.pdf",  # relative path
            }
        )
    )

    importer = MappingImporter(
        mapping_file_path=json_file,
        document_base_dir=docs_dir,
    )
    mappings = importer.import_mapping()

    assert len(mappings) == 1
    assert mappings[0].file_path == pdf_file


def test_mapping_importer_resolve_absolute_path(tmp_path):
    """Test resolving absolute paths ignores base_dir."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": str(pdf_file),  # absolute path
            }
        )
    )

    # different base dir
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    importer = MappingImporter(
        mapping_file_path=json_file,
        document_base_dir=other_dir,
    )
    mappings = importer.import_mapping()

    assert mappings[0].file_path == pdf_file


def test_mapping_importer_resolve_nonexistent_file(tmp_path):
    """Test resolving non-existent file raises FileNotFoundError."""
    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": "/nonexistent/file.pdf",
            }
        )
    )

    importer = MappingImporter(mapping_file_path=json_file)
    with pytest.raises(FileNotFoundError, match="does not exist or is not defined."):
        importer.import_mapping()


# merge_partial_paths tests
def test_merge_partial_paths_no_overlap():
    """Test merging paths with no overlap."""
    parts_a = ["Users", "docs"]
    parts_b = ["files", "test.pdf"]

    result = MappingImporter.merge_partial_paths(parts_a, parts_b)

    assert result == ["Users", "docs", "files", "test.pdf"]


def test_merge_partial_paths_full_overlap():
    """Test merging paths where b is suffix of a."""
    parts_a = ["misc", "reproduce", "hpv", "pdf"]
    parts_b = ["misc", "reproduce", "hpv", "pdf", "file.pdf"]

    result = MappingImporter.merge_partial_paths(parts_a, parts_b)

    assert result == ["misc", "reproduce", "hpv", "pdf", "file.pdf"]


# actual DocumentReferenceLinker tests
def test_linker_init_minimal():
    """Test DocumentReferenceLinker initialization with minimal args."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc])
    assert linker.documents_references == [doc]
    assert linker.document_reference_mapping is None
    assert linker.document_base_dir is None


def test_linker_init_with_mapping_list(tmp_path):
    """Test linker initialization with list of DocumentReferenceMapping."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    mapping = DocumentReferenceMapping(document_id=12345678, file_path=pdf_file)

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
    )
    assert linker.document_reference_mapping == [mapping]


def test_linker_init_with_mapping_path(tmp_path):
    """Test linker initialization with Path to mapping file."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "test.pdf"
    pdf_file.write_text("fake pdf")

    json_file = tmp_path / "mapping.json"
    json_file.write_text(
        json.dumps(
            {
                "12345678": str(pdf_file),
            }
        )
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=json_file,
    )
    assert linker.document_reference_mapping is not None
    assert len(linker.document_reference_mapping) == 1


def test_linker_init_builds_lookup_by_id():
    """Test linker builds references_by_id lookup dict."""
    citation = ReferenceFileInput(authors="Smith, John", year="2024")
    doc1 = Document(name="Doc 1", citation=citation, document_id=12345678)
    doc1.init_document_identity()
    doc2 = Document(name="Doc 2", citation=citation, document_id=87654321)
    doc2.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc1, doc2])

    assert 12345678 in linker._references_by_id
    assert 87654321 in linker._references_by_id


def test_linker_init_builds_lookup_by_author_year():
    """Test linker builds references_by_author_year lookup dicts."""
    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678,
            document_id_source=DocumentIDSource.EPPI_ITEM_ID,
            first_author="Smith",
            year="2025",
            doi=None,
        ),
    )

    linker = DocumentReferenceLinker(references=[doc])

    # should have entry keyed by author_year
    assert len(linker._references_by_author_year_longest) > 0
    assert len(linker._references_by_author_year_last) > 0


def test_linker_init_custom_strategies():
    """Test linker initialization with custom linking strategies."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    custom_strategies = [LinkingStrategy.FILENAME_ID, LinkingStrategy.MAPPING_FILE]

    linker = DocumentReferenceLinker(
        references=[doc],
        linking_strategies=custom_strategies,
    )
    assert linker.linking_strategies == custom_strategies


def test_linker_init_default_strategies():
    """Test linker uses default strategy hierarchy."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc])
    assert linker.linking_strategies == list(LinkingStrategy)


# linking factory
def test_linker_create_linking_factory_mapping_file():
    """Test _create_linking_factory returns correct function for MAPPING_FILE."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc])

    factory = linker._create_linking_factory(LinkingStrategy.MAPPING_FILE)
    assert callable(factory)
    assert factory == linker._get_linkages_mapping_file

    factory = linker._create_linking_factory(LinkingStrategy.FILENAME_AUTHOR_YEAR_LAST)
    assert callable(factory)
    assert factory.func == linker._get_linkages_filename_author_year
    assert factory.args == ("last",)

    factory = linker._create_linking_factory(
        LinkingStrategy.FILENAME_AUTHOR_YEAR_LONGEST
    )
    assert callable(factory)
    assert factory.func == linker._get_linkages_filename_author_year
    assert factory.args == ("longest",)

    factory = linker._create_linking_factory(LinkingStrategy.FILENAME_ID)
    assert callable(factory)
    assert factory == linker._get_linkages_filename_id


# mapping file
def test_linker_get_linkages_mapping_file_valid(tmp_path):
    """Test _get_linkages_mapping_file yields correct payloads."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    mapping = DocumentReferenceMapping(document_id=12345678, file_path=pdf_file)

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
    )

    results = list(linker._get_linkages_mapping_file())
    assert len(results) == 1
    assert results[0].document_id == 12345678
    assert results[0].file_path == pdf_file
    assert results[0].unlinked_document is not None


def test_linker_get_linkages_mapping_file_no_match(tmp_path):
    """Test _get_linkages_mapping_file skips unmatched mappings."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    # mapping with different ID
    mapping = DocumentReferenceMapping(document_id=99999999, file_path=pdf_file)

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
    )

    results = list(linker._get_linkages_mapping_file())
    assert len(results) == 0


def test_linker_get_linkages_mapping_file_none_mapping():
    """Test _get_linkages_mapping_file raises error when mapping is None."""
    citation = ReferenceFileInput()
    doc = Document(name="Test", citation=citation, document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=None,
    )

    with pytest.raises(TypeError, match="DocumentReferenceMapping"):
        list(linker._get_linkages_mapping_file())


def test_linker_get_linkages_mapping_file_multiple(tmp_path):
    """Test _get_linkages_mapping_file with multiple mappings."""
    pdf1 = tmp_path / "test1.pdf"
    pdf1.write_text("pdf 1")
    pdf2 = tmp_path / "test2.pdf"
    pdf2.write_text("pdf 2")

    citation = ReferenceFileInput()
    doc1 = Document(name="Doc 1", citation=citation, document_id=12345678)
    doc1.init_document_identity()
    doc2 = Document(name="Doc 2", citation=citation, document_id=87654321)
    doc2.init_document_identity()

    mappings = [
        DocumentReferenceMapping(document_id=12345678, file_path=pdf1),
        DocumentReferenceMapping(document_id=87654321, file_path=pdf2),
    ]

    linker = DocumentReferenceLinker(
        references=[doc1, doc2],
        document_reference_mapping=mappings,
    )

    results = list(linker._get_linkages_mapping_file())
    assert len(results) == 2


# mapping author-year
def test_linker_get_linkages_filename_author_year_longest(tmp_path):
    """Test _get_linkages_filename_author_year with longest strategy."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "smith_2024.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678,
            document_id_source=DocumentIDSource.EPPI_ITEM_ID,
            first_author="J Smith",
            year="2024",
            doi=None,
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_author_year("longest"))
    assert len(results) == 1
    assert results[0].file_path == pdf_file


def test_linker_get_linkages_filename_author_year_last(tmp_path):
    """Test _get_linkages_filename_author_year with last strategy."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "smith_2024.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678,
            document_id_source=DocumentIDSource.EPPI_ITEM_ID,
            first_author="J Smith",
            year="2024",
            doi=None,
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_author_year("last"))
    assert len(results) == 1


def test_linker_get_linkages_filename_author_year_no_base_dir():
    """Test _get_linkages_filename_author_year raises error without base_dir."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc])

    with pytest.raises(ValueError, match="document_base_dir"):
        list(linker._get_linkages_filename_author_year("longest"))


def test_linker_get_linkages_filename_author_year_invalid_strategy():
    """Test _get_linkages_filename_author_year raises error for invalid strategy."""
    citation = ReferenceFileInput()
    doc = Document(name="Test", citation=citation, document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=Path("/tmp"),  # noqa:S108
    )

    with pytest.raises(NotImplementedError):
        list(linker._get_linkages_filename_author_year("invalid"))


def test_linker_get_linkages_filename_author_year_skips_non_pdf_md(tmp_path):
    """Test _get_linkages_filename_author_year skips non-pdf/md files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    txt_file = docs_dir / "smith_2024.txt"
    txt_file.write_text("text file")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678,
            document_id_source=DocumentIDSource.EPPI_ITEM_ID,
            first_author="J Smith",
            year="2024",
            doi=None,
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_author_year("longest"))
    assert len(results) == 0


def test_linker_get_linkages_filename_author_year_no_match(tmp_path):
    """Test _get_linkages_filename_author_year skips non-matching files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "jones_2020.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678,
            document_id_source=DocumentIDSource.EPPI_ITEM_ID,
            first_author="J Smith",
            year="2024",
            doi=None,
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_author_year("longest"))
    assert len(results) == 0


# filename id
def test_linker_get_linkages_filename_id_valid(tmp_path):
    """Test _get_linkages_filename_id finds file named by ID."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "12345678.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_id())
    assert len(results) == 1
    assert results[0].document_id == 12345678


def test_linker_get_linkages_filename_id_no_base_dir():
    """Test _get_linkages_filename_id raises error without base_dir."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(references=[doc])

    with pytest.raises(ValueError, match="document_base_dir"):
        list(linker._get_linkages_filename_id())


def test_linker_get_linkages_filename_id_no_match(tmp_path):
    """Test _get_linkages_filename_id skips non-matching IDs."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "99999999.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_id())
    assert len(results) == 0


def test_linker_get_linkages_filename_id_md_file(tmp_path):
    """Test _get_linkages_filename_id works with markdown files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "12345678.md"
    md_file.write_text("# Markdown content")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_id())
    assert len(results) == 1
    assert results[0].format == "md"


def test_linker_get_linkages_filename_id_skips_txt(tmp_path):
    """Test _get_linkages_filename_id skips non-pdf/md files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    txt_file = docs_dir / "12345678.txt"
    txt_file.write_text("text content")

    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
    )

    results = list(linker._get_linkages_filename_id())
    assert len(results) == 0


# parse_pdf
def test_linker_parse_pdf_invalid_extension(tmp_path):
    """Test _parse_pdf raises error for non-PDF files."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("text content")

    with pytest.raises(TypeError, match="pdf"):
        DocumentReferenceLinker._parse_pdf(txt_file)


def test_linker_parse_pdf_valid(tmp_path):
    """Test _parse_pdf with valid PDF file."""
    # this requires mocking since we don't have real PDF parsing in tests
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    mock_parsed = ParsedOutput(text="fake pdf", parser_library="unknown")

    with patch("deet.processors.linker.parser") as mock_parser:
        mock_parser.return_value = mock_parsed
        result = DocumentReferenceLinker._parse_pdf(pdf_file)

    assert result == mock_parsed


def test_linker_parse_pdf_with_images(tmp_path):
    """Test _parse_pdf passes return_images flag."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    with patch("deet.processors.linker.parser") as mock_parser:
        mock_parser.return_value = ParsedOutput(
            text="fake pdf", parser_library="unknown"
        )
        DocumentReferenceLinker._parse_pdf(pdf_file, return_images=True)

    mock_parser.assert_called_once_with(
        input_=pdf_file,
        return_images=True,
        return_metadata=False,
    )


def test_linker_parse_pdf_with_metadata(tmp_path):
    """Test _parse_pdf passes return_metadata flag."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_text("fake pdf")

    with patch("deet.processors.linker.parser") as mock_parser:
        mock_parser.return_value = ParsedOutput(
            text="fake pdf", parser_library="unknown"
        )
        DocumentReferenceLinker._parse_pdf(pdf_file, return_metadata=True)

    mock_parser.assert_called_once_with(
        input_=pdf_file,
        return_images=False,
        return_metadata=True,
    )


# linking parsed document to reference
def test_linker_link_reference_parsed_document():
    """Test linking a reference with parsed document."""
    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )
    parsed = ParsedOutput(text="Parsed content", parser_library="unknown")

    result = DocumentReferenceLinker.link_reference_parsed_document(
        reference=doc,
        parsed_output=parsed,
    )

    assert result.is_linked is True
    assert result.is_final is True
    assert result.context == "Parsed content"
    assert result.context_type == ContextType.FULL_DOCUMENT
    assert result.parsed_document == parsed


def test_linker_link_reference_parsed_document_with_filepath(tmp_path):
    """Test linking with original filepath."""
    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )
    parsed = ParsedOutput(text="Parsed content", parser_library="unknown")

    original_file = tmp_path / "test.pdf"
    original_file.write_text("fake pdf")

    result = DocumentReferenceLinker.link_reference_parsed_document(
        reference=doc,
        parsed_output=parsed,
        original_filepath=original_file,
    )

    assert result.original_doc_filepath == original_file


# several doc/references
def test_linker_link_many_with_mapping_file(tmp_path):
    """Test linking multiple documents using mapping file strategy."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "test.md"
    md_file.write_text("# Markdown content")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )

    mapping = DocumentReferenceMapping(document_id=12345678, file_path=md_file)

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
        linking_strategies=[LinkingStrategy.MAPPING_FILE],
    )

    results = linker.link_many_references_parsed_documents()

    assert len(results) == 1
    assert results[0].is_linked is True
    assert results[0].context == "# Markdown content"


def test_linker_link_many_with_filename_id(tmp_path):
    """Test linking using filename ID strategy."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "12345678.md"
    md_file.write_text("# Content")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
        linking_strategies=[LinkingStrategy.FILENAME_ID],
    )

    results = linker.link_many_references_parsed_documents()

    assert len(results) == 1
    assert results[0].is_linked


def test_linker_link_many_skips_already_processed(tmp_path):
    """Test linking skips already processed documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "12345678.md"
    md_file.write_text("# Content")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )

    mapping = DocumentReferenceMapping(document_id=12345678, file_path=md_file)

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
        document_base_dir=docs_dir,
        linking_strategies=[
            LinkingStrategy.MAPPING_FILE,
            LinkingStrategy.FILENAME_ID,
        ],
    )

    results = linker.link_many_references_parsed_documents()

    # should only have 1 result, not 2 (one from each strategy)
    assert len(results) == 1
    assert results[0].is_linked


def test_linker_link_many_tries_next_strategy_on_error(tmp_path):
    """Test linking tries next strategy when one fails."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "12345678.md"
    md_file.write_text("# Content")

    citation = ReferenceFileInput(doi="10.1000/test", authors="Smith", year="2024")
    doc = Document(name="Test", citation=citation, document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=None,  # will cause MAPPING_FILE to fail
        document_base_dir=docs_dir,
        linking_strategies=[
            LinkingStrategy.MAPPING_FILE,
            LinkingStrategy.FILENAME_ID,
        ],
    )

    results = linker.link_many_references_parsed_documents()

    # should succeed via FILENAME_ID
    assert len(results) == 1


def test_linker_link_many_no_matches():
    """Test linking returns empty list when no matches found."""
    doc = Document(name="Test", citation=ReferenceFileInput(), document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        linking_strategies=[LinkingStrategy.MAPPING_FILE],
    )

    results = linker.link_many_references_parsed_documents()
    assert len(results) == 0


def test_linker_link_many_with_pdf_parsing(tmp_path):
    """Test linking parses PDF files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "12345678.pdf"
    pdf_file.write_text("fake pdf")

    doc = Document(
        name="Test",
        citation=ReferenceFileInput(),
        document_id=12345678,
        document_identity=DocumentIdentity(
            document_id=12345678, first_author="Smith", year="2024", doi="10.1000/test"
        ),
    )

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
        linking_strategies=[LinkingStrategy.FILENAME_ID],
    )

    mock_parsed = ParsedOutput(text="Parsed PDF content", parser_library="unknown")

    with patch.object(DocumentReferenceLinker, "_parse_pdf", return_value=mock_parsed):
        results = linker.link_many_references_parsed_documents()

    assert len(results) == 1
    assert results[0].context == "Parsed PDF content"


def test_linker_link_many_multiple_documents(tmp_path):
    """Test linking multiple documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md1 = docs_dir / "12345678.md"
    md1.write_text("# Doc 1")
    md2 = docs_dir / "87654321.md"
    md2.write_text("# Doc 2")

    citation = ReferenceFileInput(
        doi="10.1000/test", authors="Smith", year="2024"
    )  # see the note at the top of this file
    doc1 = Document(name="Doc 1", citation=citation, document_id=12345678)
    doc1.init_document_identity()
    doc2 = Document(name="Doc 2", citation=citation, document_id=87654321)
    doc2.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc1, doc2],
        document_base_dir=docs_dir,
        linking_strategies=[LinkingStrategy.FILENAME_ID],
    )

    results = linker.link_many_references_parsed_documents()

    assert len(results) == 2


def test_linker_link_many_partial_success(tmp_path):
    """Test linking succeeds for some documents and fails for others."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md1 = docs_dir / "12345678.md"
    md1.write_text("# Doc 1")
    # no file for doc2

    citation = ReferenceFileInput(doi="10.1000/test", authors="Smith", year="2024")
    doc1 = Document(name="Doc 1", citation=citation, document_id=12345678)
    doc1.init_document_identity()
    doc2 = Document(name="Doc 2", citation=citation, document_id=87654321)
    doc2.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc1, doc2],
        document_base_dir=docs_dir,
        linking_strategies=[LinkingStrategy.FILENAME_ID],
    )

    results = linker.link_many_references_parsed_documents()

    assert len(results) == 1
    assert results[0].document_id == 12345678


def test_linker_link_many_stops_when_all_linked(tmp_path):
    """Test linking stops trying strategies when all documents are linked."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    md_file = docs_dir / "12345678.md"
    md_file.write_text("# Content")

    citation = ReferenceFileInput(doi="10.1000/test", authors="Smith", year="2024")
    doc = Document(name="Test", citation=citation, document_id=12345678)
    doc.init_document_identity()

    mapping = DocumentReferenceMapping(document_id=12345678, file_path=md_file)

    mock_link_by_id = MagicMock()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_reference_mapping=[mapping],
        document_base_dir=docs_dir,
        linking_strategies=[
            LinkingStrategy.MAPPING_FILE,
            LinkingStrategy.FILENAME_ID,
        ],
    )

    with patch.object(linker, "_get_linkages_filename_id", mock_link_by_id):
        results = linker.link_many_references_parsed_documents()

    # FILENAME_ID should not be called since MAPPING_FILE succeeded
    mock_link_by_id.assert_not_called()
    assert len(results) == 1


def test_linker_link_many_with_images_and_metadata(tmp_path):
    """Test linking passes image and metadata flags."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    pdf_file = docs_dir / "12345678.pdf"
    pdf_file.write_text("fake pdf")

    citation = ReferenceFileInput(doi="10.1000/test", authors="Smith", year="2024")
    doc = Document(name="Test", citation=citation, document_id=12345678)
    doc.init_document_identity()

    linker = DocumentReferenceLinker(
        references=[doc],
        document_base_dir=docs_dir,
        linking_strategies=[LinkingStrategy.FILENAME_ID],
    )

    mock_parsed = ParsedOutput(text="This is parsed text.", parser_library="unknown")

    with patch.object(
        DocumentReferenceLinker,
        "_parse_pdf",
        return_value=mock_parsed,
    ) as mock_parse:
        linker.link_many_references_parsed_documents(
            return_images=True,
            return_metadata=True,
        )

    mock_parse.assert_called_once_with(
        pdf_file,
        return_images=True,
        return_metadata=True,
    )
