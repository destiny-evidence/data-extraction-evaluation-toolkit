"""Tests for identifier_utils module."""

import pytest

from deet.utils.identifier_utils import (
    MAX_DOCUMENT_ID,
    MAX_DOCUMENT_ID_DIGITS,
    MIN_DOCUMENT_ID,
    MIN_DOCUMENT_ID_DIGITS,
    hash_n_strings_to_document_id,
)


@pytest.mark.parametrize(
    "string_list",
    [
        # basic
        ["author"],
        ["author", "title"],
        ["author", "title", "year"],
        ["Smith", "John", "Cancer Research", "2024"],
        [],
        # lots
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        [str(i) for i in range(100)],
        # long
        ["Very Long Author Name Here", "Very Long Title Here"],
        ["A" * 1000],
        ["J" * 500, "K" * 500],
        # special chars
        ["test@email.com", "10.1000/journal.123"],
        ["Test@123", "Value#456", "Data$789"],
        # unicopde
        ["Müller", "José", "北京"],
        ["αβγδ", "数字", "العربية"],
        # empty/whitespace
        [""],
        ["test", "", "value", ""],
        ["", "", "", "non-empty"],
        ["  ", "\t", "\n", "test"],
        [" " * 50],
        # numbers
        ["123", "456", "789"],
        # control chars
        ["\n", "\t", "\r"],
        # mixed
        ["Smith", "2024", "10.1000/test"],
    ],
)
def test_hash_n_strings_id_always_in_valid_range(string_list):
    """Parametrized test ensuring returned IDs are always in range (4-10 digits)."""
    result = hash_n_strings_to_document_id(string_list)

    assert isinstance(result, int)

    # numerical range
    assert MIN_DOCUMENT_ID <= result <= MAX_DOCUMENT_ID

    # digit count
    digit_count = len(str(result))
    assert MIN_DOCUMENT_ID_DIGITS <= digit_count <= MAX_DOCUMENT_ID_DIGITS


def test_hash_n_strings_deterministic_output():
    """Test same input produces same output (deterministic)."""
    input_strings = ["Smith", "Cancer Research", "2024"]

    result1 = hash_n_strings_to_document_id(input_strings)
    result2 = hash_n_strings_to_document_id(input_strings)

    assert result1 == result2
