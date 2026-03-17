"""Utility functions for creating, validating and manipulating identifiers."""

import hashlib

MIN_DOCUMENT_ID_DIGITS = 4
MAX_DOCUMENT_ID_DIGITS = 10
MIN_DOCUMENT_ID = 1000
MAX_DOCUMENT_ID = 999999999


def hash_n_strings_to_eppi_id_sized_int(string_list: list[str]) -> int:
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

    # Use first 8 hex chars to select digit count (4-10)
    hash_int_1 = int(hash_hex[:8], 16)
    n_digits = (hash_int_1 % 7) + 4  # 7 possible values (4-10 inclusive)

    # Use next 8 hex chars to select value within that digit range
    hash_int_2 = int(hash_hex[8:16], 16)

    min_for_digits = 10 ** (n_digits - 1)
    max_for_digits = (10**n_digits) - 1
    range_size = max_for_digits - min_for_digits + 1

    id_ = (hash_int_2 % range_size) + min_for_digits

    # Sanity check
    id_len = len(str(id_))
    if not (MIN_DOCUMENT_ID_DIGITS <= id_len <= MAX_DOCUMENT_ID_DIGITS):
        bad_id = (
            f"id {id_} is bad, it should have between "
            f"{MIN_DOCUMENT_ID_DIGITS} and {MAX_DOCUMENT_ID_DIGITS} digits!"
        )
        raise ValueError(bad_id)

    return id_


def check_if_id_exists(new_id: int, id_list: list[int]) -> bool:
    """Check if target id (int) is in a list of ids (list[int])."""
    return new_id in id_list
