"""Unit tests for deet/data_models.py."""

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
from deet.exceptions import (
    BadDocumentIdError,
    MissingCitationElementError,
    NoAbstractError,
)
from deet.processors.parser import ParsedOutput


# DocumentIdentity stuff
def test_document_identity_creation():
    """Test creating a DocumentIdentity with all fields."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        document_id_source=DocumentIDSource.EPPI_ITEM_ID,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    assert doc_identity.document_id == 12345678
    assert doc_identity.document_id_source == DocumentIDSource.EPPI_ITEM_ID
    assert doc_identity.doi == "10.1000/test"
    assert doc_identity.first_author == "Smith"
    assert doc_identity.year == "2024"


def test_document_identity_creation_minimal():
    """Test creating a DocumentIdentity with minimal fields."""
    doc_identity = DocumentIdentity(
        doi=None,
        first_author=None,
        year=None,
    )
    assert doc_identity.document_id is None
    assert doc_identity.document_id_source is None
    assert doc_identity.doi is None


def test_document_identity_eppi_item_id_valid():
    """Test _eppi_item_id with valid 8-digit ID."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi=None,
        first_author=None,
        year=None,
    )
    result = doc_identity._eppi_item_id()
    assert result == 12345678


def test_document_identity_eppi_item_id_invalid_digits():
    """Test _eppi_item_id raises error for non-8-digit ID."""
    doc_identity = DocumentIdentity(
        document_id=1234,  # bad!
        doi=None,
        first_author=None,
        year=None,
    )
    with pytest.raises(BadDocumentIdError):
        doc_identity._eppi_item_id()


def test_document_identity_eppi_item_id_none():
    """Test _eppi_item_id raises error when document_id is None."""
    doc_identity = DocumentIdentity(
        document_id=None,
        doi=None,
        first_author=None,
        year=None,
    )
    with pytest.raises(BadDocumentIdError):
        doc_identity._eppi_item_id()


def test_document_identity_doi_id():
    """Test _doi_id creates hash from DOI."""
    doc_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author=None,
        year=None,
    )
    result = doc_identity._doi_id()
    assert isinstance(result, int)
    assert len(str(result)) == 8


def test_document_identity_doi_author_year_id():
    """Test _doi_author_year_id creates hash from multiple fields."""
    doc_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    result = doc_identity._doi_author_year_id()
    assert isinstance(result, int)
    assert len(str(result)) == 8


def test_document_identity_doi_author_year_id_missing_field():
    """Test _doi_author_year_id raises error when field is missing."""
    doc_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author=None,
        year="2024",
    )
    with pytest.raises(MissingCitationElementError):
        doc_identity._doi_author_year_id()


def test_document_identity_author_year_id():
    """Test _author_year_id creates hash from author and year."""
    doc_identity = DocumentIdentity(
        doi=None,
        first_author="Smith",
        year="2024",
    )
    result = doc_identity._author_year_id()
    assert isinstance(result, int)
    assert len(str(result)) == 8


def test_document_identity_random_int_id():
    """Test _random_int_id creates valid 8-digit integer."""
    doc_identity = DocumentIdentity(
        doi=None,
        first_author=None,
        year=None,
    )
    result = doc_identity._random_int_id()
    assert isinstance(result, int)
    assert 10000000 <= result <= 99999999
    assert len(str(result)) == 8


def test_document_identity_random_int_id_is_random():
    """Test _random_int_id produces different values."""
    doc_identity = DocumentIdentity(
        doi=None,
        first_author=None,
        year=None,
    )
    results = {doc_identity._random_int_id() for _ in range(100)}
    assert len(results) > 50


# id_factory stuff
def test_document_identity_create_id_factory_eppi():
    """Test _create_id_factory returns correct method for eppi item_id present."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    factory = doc_identity._create_id_factory(DocumentIDSource.EPPI_ITEM_ID)
    assert factory == doc_identity._eppi_item_id
    # also check it produces the right id...
    id_ = factory()
    assert id_ == doc_identity.document_id


def test_document_identity_create_id_factory_doi_author_year():
    """Test _create_id_factory returns correct method for doi_author)_year."""
    doc_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author="Naismith",
        year="2012",
    )
    factory = doc_identity._create_id_factory(DocumentIDSource.DOI_AUTHOR_YEAR)
    assert factory == doc_identity._doi_author_year_id


def test_document_identity_create_id_factory_doi():
    """Test _create_id_factory returns correct method for DOI_ID."""
    doc_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author=None,
        year=None,
    )
    factory = doc_identity._create_id_factory(DocumentIDSource.DOI_ID)
    assert factory == doc_identity._doi_id


def test_document_identity_create_id_factory_randint():
    """Test _create_id_factory returns correct method for RANDINT."""
    doc_identity = DocumentIdentity(
        doi=None,
        first_author=None,
        year=None,
    )
    factory = doc_identity._create_id_factory(DocumentIDSource.RANDINT)
    assert factory == doc_identity._random_int_id


# populate_identity()
def test_document_identity_populate_id_with_eppi():
    """Test populate_id uses EPPI_ITEM_ID when valid."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    doc_identity.populate_id()
    assert doc_identity.document_id == 12345678
    assert doc_identity.document_id_source == DocumentIDSource.EPPI_ITEM_ID


def test_document_identity_populate_id_falls_back_to_doi():
    """Test populate_id falls back to DOI when EPPI is invalid."""
    doc_identity = DocumentIdentity(
        document_id=1234,  # invalid - not 8 digits
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    doc_identity.populate_id()
    assert doc_identity.document_id is not None
    assert doc_identity.document_id_source == DocumentIDSource.DOI_AUTHOR_YEAR
    assert len(str(doc_identity.document_id)) == 8


def test_document_identity_populate_id_avoids_existing_ids():
    """Test populate_id avoids IDs in existing_ids set."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    existing_ids = {12345678}  # block the EPPI ID
    doc_identity.populate_id(existing_ids=existing_ids)
    assert doc_identity.document_id != 12345678
    assert doc_identity.document_id not in existing_ids
    assert len(str(doc_identity.document_id)) == 8


def test_document_identity_populate_id_custom_hierarchy():
    """Test populate_id respects custom hierarchy."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    doc_identity.populate_id(hierarchy=[DocumentIDSource.DOI_ID])
    assert doc_identity.document_id_source == DocumentIDSource.DOI_ID
    assert len(str(doc_identity.document_id)) == 8


def test_document_identity_populate_id_custom_bad_hierarchy():
    """Test populate_id with a custom, but bad, hierarchy."""
    doc_identity = DocumentIdentity(
        document_id=12345678,
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    with pytest.raises(KeyError):
        doc_identity.populate_id(hierarchy=["steppi_item_id"])


def test_document_identity_populate_id_adds_randint_fallback():
    """Test populate_id adds RANDINT as fallback if not in hierarchy."""
    doc_identity = DocumentIdentity(
        document_id=None,
        doi=None,
        first_author=None,
        year=None,
    )
    # no RANDINT in hierarchy
    doc_identity.populate_id(hierarchy=[DocumentIDSource.DOI_ID])
    # should still succeed via RANDINT fallback
    assert doc_identity.document_id is not None
    assert doc_identity.document_id_source == DocumentIDSource.RANDINT


def test_document_identity_hash_consistency():
    """Test that same inputs produce same hash."""
    doc_identity1 = DocumentIdentity(
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    doc_identity2 = doc_identity1.model_copy(deep=True)
    assert doc_identity1._doi_author_year_id() == doc_identity2._doi_author_year_id()


def test_populate_id_uses_different_method_seemingly_identical_records():
    """
    Test that same inputs (which might not be the same)
    don't create the same id if given a set of ids to exclude.

    NOTE: we may want some functionality that changes this
    behaviour; e.g. skips/discards seemingly identical records from a
    run.
    """
    doc_identity1 = DocumentIdentity(
        doi="10.1000/test",
        first_author="Smith",
        year="2024",
    )
    doc_identity2 = doc_identity1.model_copy(deep=True)
    existing_ids = set()
    doc_identity1.populate_id(existing_ids=existing_ids)
    existing_ids.add(doc_identity1.document_id)
    doc_identity2.populate_id(existing_ids=existing_ids)

    assert doc_identity1.document_id != doc_identity2.document_id
    assert doc_identity1.document_id_source != doc_identity2.document_id_source


# document tests
def test_document_creation():
    """Test creating a document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="This is test content",
        context_type=ContextType.FULL_DOCUMENT,
        document_id=12345678,
    )
    assert doc.name == "Test Document"
    assert doc.document_id == 12345678
    assert doc.context == "This is test content"
    assert doc.context_type == ContextType.FULL_DOCUMENT
    assert doc.is_final is False
    assert doc.is_linked is False


def test_document_creation_minimal():
    """Test creating a document with minimal required fields."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Minimal Document",
        citation=citation,
    )
    assert doc.name == "Minimal Document"
    assert doc.context is None
    assert doc.context_type == ContextType.EMPTY
    assert doc.document_id is None
    assert doc.is_final is False
    assert doc.is_linked is False


def test_document_allows_extra_fields():
    """Test that Document allows extra fields due to model_config."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        extra_field="extra value",  # type:ignore[call-arg]
        another_extra=123,
    )
    assert doc.extra_field == "extra value"  # type:ignore[attr-defined]
    assert doc.another_extra == 123  # type:ignore[attr-defined]


def test_document_is_linked_requires_context():
    """Test that is_linked=True requires context."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="context"):
        Document(
            name="Test Document",
            citation=citation,
            context=None,
            is_linked=True,
            document_identity=DocumentIdentity(
                doi="10.1000/test",
                first_author="Smith",
                year="2024",
            ),
            parsed_document=ParsedOutput(text="test", parser_library="unknown"),
        )


def test_document_is_linked_requires_context_type():
    """Test that is_linked=True requires context_type."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="context_type"):
        Document(
            name="Test Document",
            citation=citation,
            context="some context",
            context_type=None,
            is_linked=True,
            document_identity=DocumentIdentity(
                doi="10.1000/test",
                first_author="Smith",
                year="2024",
            ),
            parsed_document=ParsedOutput(text="test", parser_library="unknown"),
        )


def test_document_is_linked_requires_document_identity():
    """Test that is_linked=True requires document_identity."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="document_identity"):
        Document(
            name="Test Document",
            citation=citation,
            context="Some context",
            context_type=ContextType.FULL_DOCUMENT,
            is_linked=True,
            document_identity=None,
            parsed_document=ParsedOutput(text="test", parser_library="unknown"),
        )


def test_document_is_linked_requires_parsed_document() -> None:
    """Test that is_linked=True requires parsed_document."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="parsed_document"):
        Document(
            name="Test Document",
            citation=citation,
            context="Some context",
            is_linked=True,
            document_identity=DocumentIdentity(
                doi="10.1000/test",
                first_author="Smith",
                year="2024",
            ),
            parsed_document=None,
        )


def test_document_is_linked_valid() -> None:
    """Test that is_linked=True succeeds with all requirements met."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="Some context",
        context_type=ContextType.FULL_DOCUMENT,
        is_linked=True,
        document_identity=DocumentIdentity(
            doi="10.1000/test",
            first_author="Smith",
            year="2024",
        ),
        parsed_document=ParsedOutput(text="test", parser_library="unknown"),
    )
    assert doc.is_linked is True


def test_document_is_final_requires_context():
    """Test that is_final=True requires context."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="context.*is empty"):
        Document(
            name="Test Document",
            citation=citation,
            context=None,
            is_final=True,
            document_identity=DocumentIdentity(
                doi="10.1000/test",
                first_author="Smith",
                year="2024",
            ),
        )


def test_document_is_final_requires_good_context_type():
    """Test that is_final=True requires ContextType to not be EMPTY or None."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="context_type"):
        Document(
            name="Test Document",
            citation=citation,
            context="some context",
            context_type=ContextType.EMPTY,
            is_final=True,
            document_identity=DocumentIdentity(
                doi="10.1000/test",
                first_author="Smith",
                year="2024",
            ),
        )


def test_document_is_final_requires_document_identity():
    """Test that is_final=True requires document_identity."""
    citation = ReferenceFileInput()
    with pytest.raises(ValueError, match="document_identity"):
        Document(
            name="Test Document",
            citation=citation,
            context="some context",
            context_type=ContextType.ABSTRACT_ONLY,
            is_final=True,
            document_identity=None,
        )


def test_document_is_final_valid():
    """Test that is_final=True succeeds with all requirements met."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="Some context",
        context_type=ContextType.ABSTRACT_ONLY,
        is_final=True,
        document_identity=DocumentIdentity(
            doi="10.1000/test",
            first_author="Smith",
            year="2024",
        ),
    )
    assert doc.is_final is True


def test_document_validate_assignment():
    """Test that validation runs on assignment changes."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
        context="Some context",
        context_type=ContextType.FULL_DOCUMENT,
        document_identity=DocumentIdentity(
            doi="10.1000/test",
            first_author="Smith",
            year="2024",
        ),
        parsed_document=ParsedOutput(text="test", parser_library="unknown"),
    )
    # this will work, as conditions for linkage are met.
    doc.is_linked = True
    assert doc.is_linked is True

    # now modify something, and it will re-run the validator for is_linked and fail.
    with pytest.raises(ValueError, match="context"):
        doc.context = None


# init_document_identity()
def test_document_init_document_identity():
    """Test initializing document identity."""
    citation = ReferenceFileInput(
        doi="10.1000/test",
        authors="Smith, John",
        year="2024",
    )
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    result = doc.init_document_identity()
    assert doc.document_identity is not None
    assert result is not None
    assert isinstance(result, int)
    assert len(str(result)) == 8


def test_document_init_document_identity_with_existing_ids():
    """Test init_document_identity avoids existing IDs."""
    citation = ReferenceFileInput(
        doi="10.1000/test",
        authors="Smith, John",
        year="2024",
    )
    doc = Document(
        name="Test Document",
        citation=citation,
        document_id=12345678,
    )
    existing_ids = {12345678}
    result = doc.init_document_identity(existing_ids=existing_ids)
    assert result not in existing_ids


def test_document_init_document_identity_no_return():
    """Test init_document_identity with return_id=False."""
    citation = ReferenceFileInput(
        doi="10.1000/test",
        authors="Smith, John",
        year="2024",
    )
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    result = doc.init_document_identity(return_id=False)
    assert result is None
    assert doc.document_identity is not None


def test_document_init_document_identity_populates_document_id():
    """Test that init_document_identity populates document_id field."""
    citation = ReferenceFileInput(
        doi="10.1000/test",
        authors="Smith, John",
        year="2024",
    )
    doc = Document(
        name="Test Document",
        citation=citation,
        document_id=None,
    )
    doc.init_document_identity()
    assert doc.document_id is not None
    assert doc.document_id == doc.document_identity.document_id


def test_document_author_year_from_document_identity_longest():
    """Test generating author_year string from document identity."""
    doc = Document(
        name="Test Document",
        citation=ReferenceFileInput(),
    )
    doc.document_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author="Mary Watson-Parker",
        year="2024",
    )
    result = doc.author_year_from_document_identity(substring_strategy="longest")
    assert result == "watson-parker_2024"


def test_document_author_year_from_document_identity_last():
    """Test author_year with multi-word author name uses longest component."""
    doc = Document(
        name="Test Document",
        citation=ReferenceFileInput(),
    )
    doc.document_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author="MaryJeanetteAndrea Watson-Parker",
        year="2024",
    )
    result = doc.author_year_from_document_identity(substring_strategy="last")
    assert "watson-parker_2024" in result


def test_document_author_year_raises_when_missing_data():
    """Test author_year raises ValueError when required data is missing."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    doc.document_identity = DocumentIdentity(
        doi="10.1000/test",
        first_author=None,
        year="2024",
    )
    with pytest.raises(ValueError):  # noqa: PT011
        doc.author_year_from_document_identity(substring_strategy="longest")


# set_abstract_context
def test_document_set_abstract_context():
    """Test setting abstract as context."""
    citation = ReferenceFileInput(
        abstract="This is the abstract text.",
    )
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    mock_labs_ref = MagicMock()
    mock_labs_ref.abstract = "This is the abstract text."

    with patch("deet.data_models.documents.LabsReference", return_value=mock_labs_ref):
        doc.set_abstract_context()

    assert doc.context == "This is the abstract text."
    assert doc.context_type == ContextType.ABSTRACT_ONLY


def test_document_set_abstract_context_no_abstract():
    """Test set_abstract_context raises NoAbstractError when no abstract."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    with pytest.raises(NoAbstractError):
        doc.set_abstract_context()


# link_parsed_document()
def test_document_link_parsed_document():
    """Test linking a parsed document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    parsed = ParsedOutput(text="Parsed text content", parser_library="unknown")
    doc.link_parsed_document(parsed)
    assert doc.parsed_document == parsed
    assert doc.original_doc_filepath is None


def test_document_link_parsed_document_with_filepath(tmp_path: Path):
    """Test linking parsed document with valid filepath."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )

    test_file = tmp_path / "test.pdf"
    test_file.write_text("test content")

    parsed = ParsedOutput(text="Parsed text content", parser_library="unknown")
    doc.link_parsed_document(parsed, original_doc_filepath=test_file)
    assert doc.parsed_document == parsed
    assert doc.original_doc_filepath == test_file


def test_document_link_parsed_document_invalid_filepath(tmp_path: Path):
    """Test linking parsed document with invalid filepath sets None."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )

    nonexistent_file = tmp_path / "nonexistent.pdf"
    parsed = ParsedOutput(text="Parsed text content", parser_library="unknown")
    doc.link_parsed_document(parsed, original_doc_filepath=nonexistent_file)
    assert doc.parsed_document == parsed
    assert doc.original_doc_filepath is None


# context_from_parsed (symlinking context)
def test_document_set_context_from_parsed():
    """Test setting context from parsed document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    doc.parsed_document = ParsedOutput(
        text="The parsed text content", parser_library="unknown"
    )
    doc.set_context_from_parsed()
    assert doc.context == "The parsed text content"
    assert doc.context_type == ContextType.FULL_DOCUMENT


def test_document_set_context_from_parsed_no_parsed_doc():
    """Test set_context_from_parsed with no parsed_document."""
    citation = ReferenceFileInput()
    doc = Document(
        name="Test Document",
        citation=citation,
    )
    doc.set_context_from_parsed()
    assert doc.context is None


# save/load methods
