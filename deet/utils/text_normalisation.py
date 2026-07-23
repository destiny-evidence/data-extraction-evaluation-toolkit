"""Shared text normalisation helpers for evaluation and EPPI citation parsing."""

import re
from typing import Any

# Residual EPPI highlight markers that may remain after parsing.
_EPPI_MARKUP_PATTERN = re.compile(r"\[¬[se]\]")
# Leading/trailing quote characters (straight and curly).
_SURROUNDING_QUOTES = frozenset("\"'`\u201c\u201d\u2018\u2019")


def normalize_whitespace(text: str) -> str:
    """
    Collapse runs of whitespace to a single space and strip ends.

    Args:
        text: Input string (may contain newlines, tabs, or repeated spaces).

    Returns:
        String with internal whitespace collapsed and leading/trailing whitespace
        removed. Empty input yields ``""``.

    """
    return re.sub(r"\s+", " ", text).strip()


def clean_extracted_text(text: str) -> str:
    """
    Clean an already-parsed highlight or citation fragment.

    Removes residual EPPI markup markers (``[¬s]`` / ``[¬e]``), strips surrounding
    quotes, and normalises whitespace. Intended for fragments extracted *after*
    markup-aware parsing — not for raw EPPI ``Text`` fields.

    Args:
        text: Highlight or citation fragment (already separated from page/markup
            structure).

    Returns:
        Cleaned string ready for display or comparison.

    """
    cleaned = _EPPI_MARKUP_PATTERN.sub("", text)
    cleaned = normalize_whitespace(cleaned)
    while cleaned and cleaned[0] in _SURROUNDING_QUOTES:
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in _SURROUNDING_QUOTES:
        cleaned = cleaned[:-1]
    return normalize_whitespace(cleaned)


def normalize_string_for_match(text: str, *, case_insensitive: bool = True) -> str:
    """
    Normalise a string for comparison or search.

    Applies :func:`normalize_whitespace` and, when ``case_insensitive`` is True
    (the default), lowercases the result.

    Args:
        text: Input string to normalise.
        case_insensitive: If True, lowercase after whitespace normalisation.

    Returns:
        Normalised string suitable for matching.

    """
    normalised = normalize_whitespace(text)
    if case_insensitive:
        return normalised.lower()
    return normalised


def normalize_list_elements(items: list[Any]) -> list[Any]:
    """
    Apply :func:`normalize_string_for_match` to each string element.

    Non-string elements are passed through unchanged.

    Args:
        items: Mixed list of values (typically strings plus other types).

    Returns:
        New list with string elements normalised (case-insensitive by default).

    """
    return [
        normalize_string_for_match(item) if isinstance(item, str) else item
        for item in items
    ]
