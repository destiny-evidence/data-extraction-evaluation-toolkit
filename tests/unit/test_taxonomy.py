"""Tests for consuming, traversing, and linking published vocabularies."""

from pathlib import Path

import pytest

from deet.data_models.taxonomy import ConceptScheme, load_schemes_from_ttl

VOCAB_PATH = Path(
    "tests/test_files/vocabularies/small-test-taxonomy-0-1-initial-release.ttl"
)


@pytest.fixture
def schemes() -> list[ConceptScheme]:
    return load_schemes_from_ttl(VOCAB_PATH)


def test_from_ttl_happy_path(schemes):
    # This should contain 1 scheme
    assert len(schemes) == 1
    roots = schemes[0].roots
    assert len(roots) == 2
    concept_titles = [concept.pref_label for concept in roots]
    assert "Infectious diseases" in concept_titles
