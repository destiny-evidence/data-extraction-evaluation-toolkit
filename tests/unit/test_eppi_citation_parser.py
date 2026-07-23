"""Unit tests for EPPI citation markup parsing."""

from deet.data_models.eppi import EppiItemAttributeFullTextDetails
from deet.processors.eppi_citation_parser import (
    format_parsed_citations,
    parse_eppi_citation_text,
    parse_eppi_citations_from_details,
)


def test_parse_standard_page_and_highlight() -> None:
    """Parse ``Page N:`` prefix and closed highlight markers."""
    raw = 'Page 7:\n[¬s]"Odds ratio[¬e]"'
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page == 7
    assert parsed.highlight_text == "Odds ratio"
    assert parsed.raw == raw


def test_parse_conftest_fixture_markup() -> None:
    """Parse the sample markup used in ``tests/conftest.py``."""
    raw = 'Page 1:\n[¬s]"Dolor si amet...[¬e]"'
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page == 1
    assert parsed.highlight_text == "Dolor si amet..."
    assert parsed.raw == raw


def test_parse_multi_highlight_joins_with_space() -> None:
    """Join multiple closed highlights within one Text block with a space."""
    raw = 'Page 7:\n[¬s]"A"[¬e]\n[¬s]"B"[¬e]'
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page == 7
    assert parsed.highlight_text == "A B"


def test_parse_missing_page() -> None:
    """Missing ``Page N:`` yields ``page=None`` and cleaned highlight."""
    raw = '[¬s]"No page here"[¬e]'
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page is None
    assert parsed.highlight_text == "No page here"


def test_parse_plain_text_without_markup() -> None:
    """Plain text with no markup becomes cleaned highlight; page stays None."""
    raw = "Just plain text, no markup"
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page is None
    assert parsed.highlight_text == "Just plain text, no markup"


def test_parse_plain_text_with_page_prefix() -> None:
    """``Page N:`` without highlight markers still extracts page and remainder."""
    raw = "Page 3:\nJust plain remainder"
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page == 3
    assert parsed.highlight_text == "Just plain remainder"


def test_parse_malformed_unclosed_highlight() -> None:
    """Unclosed ``[¬s]`` takes the remainder as the highlight (best-effort)."""
    raw = 'Page 5:\n[¬s]"Broken'
    parsed = parse_eppi_citation_text(raw)
    assert parsed.page == 5
    assert parsed.highlight_text == "Broken"


def test_parse_empty_string() -> None:
    """Empty input yields empty highlight and no page."""
    parsed = parse_eppi_citation_text("")
    assert parsed.page is None
    assert parsed.highlight_text == ""
    assert parsed.raw == ""


def test_parse_case_insensitive_page_prefix() -> None:
    """``page`` prefix matching is case-insensitive."""
    parsed = parse_eppi_citation_text('page 12:\n[¬s]"Title"[¬e]')
    assert parsed.page == 12
    assert parsed.highlight_text == "Title"


def test_parse_eppi_citations_from_details() -> None:
    """Parse each detail entry independently; skip empty text."""
    details = [
        EppiItemAttributeFullTextDetails(
            item_document_id=1,
            text='Page 3:\n[¬s]"First"[¬e]',
        ),
        EppiItemAttributeFullTextDetails(
            item_document_id=2,
            text='Page 7:\n[¬s]"Second"[¬e]',
        ),
        EppiItemAttributeFullTextDetails(
            item_document_id=3,
            text="",
        ),
    ]
    parsed = parse_eppi_citations_from_details(details)
    assert len(parsed) == 2
    assert parsed[0].page == 3
    assert parsed[0].highlight_text == "First"
    assert parsed[1].page == 7
    assert parsed[1].highlight_text == "Second"


def test_parse_eppi_citations_from_details_none_or_empty() -> None:
    """``None`` or empty details yield an empty list."""
    assert parse_eppi_citations_from_details(None) == []
    assert parse_eppi_citations_from_details([]) == []


def test_format_parsed_citations_joins_with_colon() -> None:
    """Join multiple citations with ``: `` for CSV columns."""
    citations = parse_eppi_citations_from_details(
        [
            EppiItemAttributeFullTextDetails(
                text='Page 3:\n[¬s]"First highlight"[¬e]',
            ),
            EppiItemAttributeFullTextDetails(
                text='Page 7:\n[¬s]"Second highlight"[¬e]',
            ),
        ]
    )
    citation_page, citation_highlight = format_parsed_citations(citations)
    assert citation_page == "3: 7"
    assert citation_highlight == "First highlight: Second highlight"


def test_format_parsed_citations_omits_missing_pages() -> None:
    """Omit missing pages from the joined page string."""
    citations = [
        parse_eppi_citation_text('[¬s]"Only highlight"[¬e]'),
        parse_eppi_citation_text('Page 2:\n[¬s]"With page"[¬e]'),
    ]
    citation_page, citation_highlight = format_parsed_citations(citations)
    assert citation_page == "2"
    assert citation_highlight == "Only highlight: With page"
