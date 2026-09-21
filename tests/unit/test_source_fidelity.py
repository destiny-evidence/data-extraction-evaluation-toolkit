"""Unit tests for source-fidelity helpers."""

from deet.data_models.base import AttributeType
from deet.evaluators.source_fidelity import (
    classify_match_status,
    is_gold_value_in_text,
    parse_numeric_tokens,
)


def test_parse_numeric_tokens_handles_decimal_variants() -> None:
    """Numeric token parsing captures equivalent decimal representations."""
    values = parse_numeric_tokens("Value list: 0.05, .05, 0.050 and 11 and 1.11.")
    assert 0.05 in values
    assert 11.0 in values
    assert 1.11 in values


def test_numeric_gold_value_in_text_avoids_digit_substring_confusion() -> None:
    """Gold 1.11 is not confused with 0.11 or 11."""
    assert not is_gold_value_in_text(
        gold_value=1.11,
        haystack_text="Reported as 0.11 with 95% CI.",
        attribute_type=AttributeType.FLOAT,
        edit_distance_threshold=0.9,
        allow_string_near_match=False,
    )
    assert is_gold_value_in_text(
        gold_value=0.05,
        haystack_text="The estimate is .05 across studies.",
        attribute_type=AttributeType.FLOAT,
        edit_distance_threshold=0.9,
        allow_string_near_match=False,
    )


def test_string_gold_value_uses_normalized_search() -> None:
    """Whitespace/case differences still count as present."""
    assert is_gold_value_in_text(
        gold_value="Odds  Ratio",
        haystack_text="odds ratio",
        attribute_type=AttributeType.STRING,
        edit_distance_threshold=0.9,
        allow_string_near_match=False,
    )


def test_match_status_for_good_and_bad_source() -> None:
    """Match status distinguishes bad-source from good-source extraction errors."""
    assert (
        classify_match_status(
            gold_value=1.11,
            predicted_value=0.11,
            gold_in_context=False,
            attribute_type=AttributeType.FLOAT,
            edit_distance_threshold=0.9,
        )
        == "extraction_error_bad_source"
    )
    assert (
        classify_match_status(
            gold_value="hypertension",
            predicted_value="hypotension",
            gold_in_context=True,
            attribute_type=AttributeType.STRING,
            edit_distance_threshold=0.9,
        )
        == "extraction_error_good_source"
    )
    assert (
        classify_match_status(
            gold_value="32",
            predicted_value="32",
            gold_in_context=True,
            attribute_type=AttributeType.STRING,
            edit_distance_threshold=0.9,
        )
        == "exact_match"
    )
    assert (
        classify_match_status(
            gold_value="hypertension",
            predicted_value=None,
            gold_in_context=True,
            attribute_type=AttributeType.STRING,
            edit_distance_threshold=0.9,
        )
        == "missing_prediction"
    )
