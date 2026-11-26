"""Tests for consuming, traversing, and linking published vocabularies."""

import json
from pathlib import Path

import pytest

from deet.data_models.base import Attribute, AttributeType
from deet.data_models.taxonomy import (
    Concept,
    ConceptMappingRow,
    ConceptScheme,
    load_schemes_from_ttl,
)

VOCAB_PATH = Path(
    "tests/test_files/vocabularies/small-test-taxonomy-0-1-initial-release.ttl"
)


@pytest.fixture
def schemes() -> list[ConceptScheme]:
    return load_schemes_from_ttl(VOCAB_PATH)


class TestLoadFromTtl:
    """Tests for ttl loading behaviour."""

    def test_happy_path(self, schemes):
        """Test that the taxonomy parses correctly."""
        assert len(schemes) == 1
        roots = schemes[0].roots
        assert len(roots) == 2
        concept_titles = [concept.pref_label for concept in roots]
        assert "Infectious diseases" in concept_titles


class TestBuildPrompt:
    """Tests for building prompts from taxonomy concepts."""

    def test_definition_only(self, schemes):
        """Test that prompts can be built from a definition."""
        infectious = schemes[0].get_concept("MINT000003")
        assert infectious.build_prompt(["definition"]) == (
            "Diseases caused by pathogenic microorganisms"
            " transmitted between humans or from animals."
        )

    def test_definition_and_scope_note(self, schemes):
        """Test that definition and scope note are joined with a newline."""
        infectious = schemes[0].get_concept("MINT000003")
        assert infectious.build_prompt(["definition", "scope_note"]) == (
            "Diseases caused by pathogenic microorganisms"
            " transmitted between humans or from animals.\n"
            " Includes vector-borne, waterborne, and airborne infectious diseases."
        )

    def test_alt_labels(self, schemes):
        """Test that alt labels are joined with semicolons."""
        malaria = schemes[0].get_concept("MINT000004")
        assert malaria.build_prompt(["alt_labels"]) == (
            "Plasmodium infection; mosquito-borne fever"
        )

    def test_falls_back_to_pref_label(self, schemes):
        """Test that pref_label is used when requested fields yield no text."""
        malaria = schemes[0].get_concept("MINT000004")
        assert malaria.build_prompt(["definition"]) == "Malaria"

    def test_raises_when_no_text(self):
        """Test that ValueError is raised when no text can be found for any field."""
        c = Concept(
            identifier="empty",
            uri="https://example.org/empty",
            pref_label="",
            broader=[],
            alt_labels=[],
        )
        with pytest.raises(ValueError, match="No prompt text"):
            c.build_prompt(["definition", "scope_note"])


class TestTraversal:
    """Test methods to traverse concept hierarchy."""

    def test_narrower_returns_children(self, schemes):
        """Test that narrower returns direct children of a concept."""
        children = schemes[0].narrower("MINT000003")
        labels = {c.pref_label for c in children}
        assert labels == {"Malaria", "Dengue"}

    def test_narrower_leaf_returns_empty(self, schemes):
        """Test that a leaf concept has no narrower concepts."""
        assert schemes[0].narrower("MINT000004") == []

    def test_get_concept_absent_returns_none(self, schemes):
        """Test that get_concept returns None for an unknown id."""
        assert schemes[0].get_concept("MINT999999") is None


class TestMapConcepts:
    """Tests for mapping taxonomy concepts to deet attributes."""

    def test_mapped_concept_gets_attribute(self, schemes, tmp_path):
        """Test that a concept in the mapping file gets the corresponding attribute."""
        mapping = [
            ConceptMappingRow(
                attribute_name="Infectious diseases", concept_id="MINT000003"
            )
        ]
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(json.dumps([row.model_dump() for row in mapping]))

        attr = Attribute(
            attribute_id=1,
            attribute_label="Infectious diseases",
            output_data_type=AttributeType.BOOL,
        )
        mapped = schemes[0].map_concepts(mapping_file=mapping_file, attributes=[attr])
        assert mapped.get_concept("MINT000003").attribute.attribute_id == 1

    def test_unmapped_concept_gets_synthetic_attribute(self, schemes, tmp_path):
        """Concepts not in mapping file get a synthetic attribute from pref_label."""
        mapping_file = tmp_path / "mapping.json"
        mapping_file.write_text(json.dumps([]))

        mapped = schemes[0].map_concepts(mapping_file=mapping_file, attributes=[])
        assert mapped.get_concept("MINT000004").attribute.attribute_label == "Malaria"

    def test_raises_without_mapping_file(self, schemes):
        """Mapping without a file is not yet supported."""
        with pytest.raises(NotImplementedError):
            schemes[0].map_concepts(mapping_file=None, attributes=[])
