"""Utility functions for creating, validating and manipulating identifiers."""

import hashlib

DOCUMENT_ID_N_DIGITS = 8
MIN_DOCUMENT_ID = 10000000
MAX_DOCUMENT_ID = 99999999


def hash_n_strings_to_eight_digit_int(string_list: list[str]) -> int:
    """
    Convert n strings into an 8-digit integer using hash-based combination.

    Args:
        string_list: a list of strings to hash.

    Returns:
        An 8-digit integer (10000000 to 99999999)

    """
    combined = "|".join(string_list)

    hash_object = hashlib.sha256(combined.encode())
    hash_hex = hash_object.hexdigest()

    # convert first 8 hex chars to integer and map to 8-digit range
    hash_int = int(hash_hex[:8], 16)

    return (hash_int % MAX_DOCUMENT_ID) + MIN_DOCUMENT_ID


def check_if_id_exists(new_id: int, id_list: list[int]) -> bool:
    """Check if target id (int) is in a list of ids (list[int])."""
    return new_id in id_list
