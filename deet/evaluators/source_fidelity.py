"""Helpers for source-fidelity checks and row-level match status."""

import re

from rapidfuzz.distance import Levenshtein

from deet.data_models.base import AttributeType
from deet.utils.text_normalisation import normalize_string_for_match

_NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![0-9])(?:[0-9]+\.[0-9]+|[0-9]+|\.[0-9]+)(?![0-9])"
)


def parse_numeric_tokens(text: str | None) -> list[float]:
    """
    Parse standalone numeric tokens from free text.

    Args:
        text: Source text to scan.

    Returns:
        List of parsed float values. Invalid tokens are ignored.

    """
    values: list[float] = []
    if not text:
        return values
    for token in _NUMERIC_TOKEN_PATTERN.findall(text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _string_near_match_in_haystack(
    gold_text: str, haystack_text: str, threshold: float
) -> bool:
    """
    Check approximate string presence via n-gram window similarity.

    Args:
        gold_text: Normalised gold text.
        haystack_text: Normalised haystack.
        threshold: Minimum normalised similarity in ``[0, 1]``.

    Returns:
        True if any equal-length n-gram window reaches ``threshold``.

    """
    gold_tokens = gold_text.split()
    haystack_tokens = haystack_text.split()
    if not gold_tokens or len(gold_tokens) > len(haystack_tokens):
        return False
    window_size = len(gold_tokens)
    for start in range(len(haystack_tokens) - window_size + 1):
        window = " ".join(haystack_tokens[start : start + window_size])
        if Levenshtein.normalized_similarity(gold_text, window) >= threshold:
            return True
    return False


def is_gold_value_in_text(
    *,
    gold_value: object,
    haystack_text: str | None,
    attribute_type: AttributeType,
    edit_distance_threshold: float,
    allow_string_near_match: bool,
) -> bool:
    """
    Check whether a gold value can be found in source text.

    Args:
        gold_value: Gold-standard output_data.
        haystack_text: Citation/context text to search.
        attribute_type: Attribute output type.
        edit_distance_threshold: Similarity threshold used for optional
            string near-match.
        allow_string_near_match: Whether approximate match fallback is enabled.

    Returns:
        True when the gold value is considered present in the haystack.

    """
    if haystack_text is None or haystack_text.strip() == "":
        return False

    if attribute_type == AttributeType.STRING:
        gold_text = normalize_string_for_match(str(gold_value))
        haystack_normalised = normalize_string_for_match(haystack_text)
        if not gold_text:
            present = False
        elif gold_text in haystack_normalised:
            present = True
        elif allow_string_near_match:
            present = _string_near_match_in_haystack(
                gold_text, haystack_normalised, edit_distance_threshold
            )
        else:
            present = False
        return present

    if attribute_type in {AttributeType.INTEGER, AttributeType.FLOAT}:
        try:
            gold_float = float(str(gold_value))
        except (TypeError, ValueError):
            return False
        return any(value == gold_float for value in parse_numeric_tokens(haystack_text))

    return False


def classify_match_status(
    *,
    gold_value: object,
    predicted_value: object | None,
    gold_in_context: bool,
    attribute_type: AttributeType,
    edit_distance_threshold: float,
) -> str | None:
    """
    Classify row-level match status for STRING / INTEGER / FLOAT attributes.

    Args:
        gold_value: Gold-standard value.
        predicted_value: Model predicted value.
        gold_in_context: Whether gold is found in parsed context.
        attribute_type: Attribute output type.
        edit_distance_threshold: Similarity threshold for STRING near-match.

    Returns:
        One of ``exact_match``, ``near_match``, ``missing_prediction``,
        ``extraction_error_bad_source``, or ``extraction_error_good_source``
        for STRING/INTEGER/FLOAT, else ``None``.

    """
    if attribute_type not in {
        AttributeType.STRING,
        AttributeType.INTEGER,
        AttributeType.FLOAT,
    }:
        return None
    if predicted_value is None:
        return "missing_prediction"
    if predicted_value == gold_value:
        return "exact_match"
    if attribute_type == AttributeType.STRING:
        gold_text = normalize_string_for_match(str(gold_value))
        pred_text = normalize_string_for_match(str(predicted_value))
        similarity = Levenshtein.normalized_similarity(gold_text, pred_text)
        if similarity >= edit_distance_threshold:
            return "near_match"
    if not gold_in_context:
        return "extraction_error_bad_source"
    return "extraction_error_good_source"
