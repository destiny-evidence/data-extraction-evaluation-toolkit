"""Unit tests for shared text normalisation helpers."""

from deet.utils.text_normalisation import (
    clean_extracted_text,
    normalize_list_elements,
    normalize_string_for_match,
    normalize_whitespace,
)


def test_normalize_whitespace_collapses_runs_and_strips() -> None:
    """Collapse internal whitespace and strip ends."""
    assert normalize_whitespace("  Odds   ratio  ") == "Odds ratio"
    assert normalize_whitespace("Page 7:\n\n[¬s]...") == "Page 7: [¬s]..."
    assert normalize_whitespace("") == ""
    assert normalize_whitespace("   \n\t  ") == ""


def test_clean_extracted_text_removes_quotes_and_residual_markup() -> None:
    """Strip surrounding quotes and residual EPPI markers after parsing."""
    assert clean_extracted_text("Odds ratio") == "Odds ratio"
    assert clean_extracted_text('"Odds ratio"') == "Odds ratio"
    assert clean_extracted_text("[¬s]Odds ratio[¬e]") == "Odds ratio"
    assert clean_extracted_text('  "Odds   ratio"  ') == "Odds ratio"
    assert clean_extracted_text("\u201cOdds ratio\u201d") == "Odds ratio"
    assert clean_extracted_text("'Odds ratio'") == "Odds ratio"


def test_normalize_string_for_match_case_folding() -> None:
    """Default lowercases; case_insensitive=False preserves case."""
    assert normalize_string_for_match("  Odds   Ratio  ") == "odds ratio"
    assert normalize_string_for_match("ODDS RATIO") == "odds ratio"
    assert (
        normalize_string_for_match("  Odds   Ratio  ", case_insensitive=False)
        == "Odds Ratio"
    )


def test_normalize_list_elements_normalises_strings_only() -> None:
    """Normalise string elements; pass non-strings through unchanged."""
    assert normalize_list_elements(["  Foo ", "BAR  ", 42]) == ["foo", "bar", 42]
    assert normalize_list_elements([]) == []
    assert normalize_list_elements([None, True]) == [None, True]
