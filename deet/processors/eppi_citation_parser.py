"""Parse raw EPPI citation markup into page and highlight fields."""

import re
from dataclasses import dataclass

from deet.data_models.eppi import EppiItemAttributeFullTextDetails
from deet.utils.text_normalisation import clean_extracted_text

# ``Page 7:`` (case-insensitive); captures the page number.
_PAGE_PATTERN = re.compile(r"Page\s+(\d+)\s*:", re.IGNORECASE)
# Highlight span between EPPI start/end markers; markers are structure — do not
# strip them before matching.
_HIGHLIGHT_PATTERN = re.compile(r"\[¬s\](.*?)\[¬e\]", re.DOTALL)
# Unclosed highlight: ``[¬s]`` present but no ``[¬e]`` — take remainder of text.
_UNCLOSED_HIGHLIGHT_PATTERN = re.compile(r"\[¬s\](.*)", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ParsedEppiCitation:
    """
    Structured fields extracted from a raw EPPI ``Text`` citation block.

    Attributes:
        page: Page number from a ``Page N:`` prefix, or ``None`` if absent/invalid.
        highlight_text: Cleaned highlight content (quotes/residual markup removed).
        raw: Original input string retained for audit.

    """

    page: int | None
    highlight_text: str
    raw: str


def parse_eppi_citation_text(raw_text: str) -> ParsedEppiCitation:
    """
    Parse raw EPPI ``item_attribute_full_text_details`` markup.

    Operates on **raw** markup so ``[¬s]`` / ``[¬e]`` can be used as structure.
    Extracted highlight fragments are cleaned via :func:`clean_extracted_text`.

    Example::

        Page 7:
        [¬s]"Odds ratio[¬e]"

    yields ``page=7``, ``highlight_text="Odds ratio"``.

    Args:
        raw_text: Raw EPPI ``Text`` field value.

    Returns:
        :class:`ParsedEppiCitation` with best-effort page and highlight fields.

    """
    raw = raw_text or ""
    page = _extract_page(raw)
    highlight_text = _extract_highlight(raw)
    return ParsedEppiCitation(page=page, highlight_text=highlight_text, raw=raw)


def parse_eppi_citations_from_details(
    details: list[EppiItemAttributeFullTextDetails] | None,
) -> list[ParsedEppiCitation]:
    """
    Parse each EPPI full-text detail entry independently.

    Args:
        details: List of :class:`EppiItemAttributeFullTextDetails`, or ``None``.

    Returns:
        One :class:`ParsedEppiCitation` per detail with a non-empty ``text``.
        Empty or ``None`` input yields ``[]``.

    """
    if not details:
        return []
    parsed: list[ParsedEppiCitation] = []
    for detail in details:
        text = detail.text
        if text is not None and str(text).strip():
            parsed.append(parse_eppi_citation_text(str(text)))
    return parsed


def format_parsed_citations(
    citations: list[ParsedEppiCitation],
) -> tuple[str, str]:
    """
    Join multiple parsed citations for CSV export.

    Uses ``": "`` as the separator (same convention as the raw
    ``item_attribute_full_text_details`` column).

    Args:
        citations: Parsed citation list (may be empty).

    Returns:
        ``(citation_page, citation_highlight_text)`` strings. Missing pages are
        omitted from the page string; empty highlights are omitted from the
        highlight string.

    """
    pages = [str(c.page) for c in citations if c.page is not None]
    highlights = [c.highlight_text for c in citations if c.highlight_text]
    return ": ".join(pages), ": ".join(highlights)


def _extract_page(raw: str) -> int | None:
    """Return the first ``Page N:`` number in ``raw``, or ``None``."""
    match = _PAGE_PATTERN.search(raw)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_highlight(raw: str) -> str:
    """
    Extract and clean highlight text from EPPI markup.

    Prefer closed ``[¬s]...[¬e]`` spans (joined with a space when several exist).
    Fall back to unclosed ``[¬s]...`` remainder, then to plain text with the page
    prefix removed when no markup is present.
    """
    closed = _HIGHLIGHT_PATTERN.findall(raw)
    if closed:
        cleaned_parts = [clean_extracted_text(part) for part in closed]
        return " ".join(part for part in cleaned_parts if part)

    unclosed = _UNCLOSED_HIGHLIGHT_PATTERN.search(raw)
    if unclosed is not None:
        return clean_extracted_text(unclosed.group(1))

    # Plain text: strip a leading ``Page N:`` prefix if present, then clean.
    without_page = _PAGE_PATTERN.sub("", raw, count=1)
    return clean_extracted_text(without_page)
