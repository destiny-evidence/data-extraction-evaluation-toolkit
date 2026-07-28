"""
Data models for consuming hierarchical vocabularies.

These models are loosely based on [`destiny-evidence/taxonomy-builder`](https://github.com/destiny-evidence/taxonomy-builder)

implementing only those methods required for the initial case (see integration tests)
of a nested set of concepts in concept schemes.

TODO: Extend to cover classes and properties.
"""

from pathlib import Path

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter
from rdflib import SKOS, Graph, URIRef

from deet.data_models.base import Attribute
from deet.utils._vendor.taxonomy_builder.rdf_parser import (
    analyze_graph,
    detect_format,
    get_concept_pref_label,
    get_identifier_from_uri,
    get_scheme_title,
    parse_rdf,
)


class Concept(BaseModel):
    """A single SKOS concept within a scheme."""

    model_config = ConfigDict(extra="ignore")

    pref_label: str = Field(description="Preferred label.")
    identifier: str = Field(description="URI-safe identifier segment.")
    uri: str = Field(description="Full concept URI.")
    definition: str | None = Field(default=None, description="Concept definition.")
    scope_note: str | None = Field(
        default=None, description="Scope note clarifying intended use."
    )
    alt_labels: list[str] = Field(
        default_factory=list,
        description="Alternative labels (synonyms, abbreviations).",
    )
    broader: list[str] = Field(
        default_factory=list, description="uris of broader (parent) concepts."
    )
    related: list[str] = Field(
        default_factory=list, description="uris of related concepts."
    )
    type_uris: list[str] = Field(
        default_factory=list,
        description="OWL class type URIs beyond skos:Concept.",
    )

    @classmethod
    def from_graph(cls, g: "Graph", concept_uri: "URIRef") -> "tuple[str, Concept]":
        """Load from parsed graph."""
        pref_label, _ = get_concept_pref_label(g, concept_uri)
        return get_identifier_from_uri(concept_uri), cls(
            uri=str(concept_uri),
            identifier=get_identifier_from_uri(concept_uri),
            pref_label=pref_label,
            alt_labels=[str(lbl) for lbl in g.objects(concept_uri, SKOS.altLabel)],
            definition=next(
                (str(d) for d in g.objects(concept_uri, SKOS.definition)),
                None,
            ),
            scope_note=next(
                (str(s) for s in g.objects(concept_uri, SKOS.scopeNote)),
                None,
            ),
            broader=[str(b) for b in g.objects(concept_uri, SKOS.broader)],
        )


class ConceptMappingRow(BaseModel):
    """Defines the expected shape of a mapping from attribute labels to concept_ids."""

    model_config = ConfigDict(extra="ignore")
    attribute_name: str = Field(
        validation_alias=AliasChoices("attribute_name", "col_pipe")
    )
    concept_id: str


class MappedConcept(Concept):
    """A concept that has been mapped to a deet Attribute."""

    attribute: Attribute


class ConceptScheme[C: Concept](BaseModel):
    """
    A concept scheme: one taxonomy "field" owning a flat map of concepts.

    The scheme's ``broader`` edges define the hierarchy; ``narrower`` and
    descendant traversal are derived here.
    """

    model_config = ConfigDict(extra="ignore")

    title: str = Field(description="Vocabulary title.")
    description: str | None = Field(default=None, description="Vocabulary description.")
    uri: str = Field(description="SKOS ConceptScheme URI.")
    top_concepts: list[str] = Field(
        default_factory=list, description="concept uris of root concepts (no broader)."
    )
    concepts: dict[str, C] = Field(
        default_factory=dict, description="Concepts keyed by uri."
    )

    def get_concept(self, concept_id: str) -> Concept | None:
        """Return the concept with ``concept_id``, or None if absent."""
        return self.concepts.get(concept_id)

    @property
    def roots(self) -> list[Concept]:
        """Return the concepts at the root of the scheme."""
        return [c for uid in self.top_concepts if (c := self.get_concept(uid))]

    def _narrower_index(self) -> dict[str, list[str]]:
        """Build the inverse of ``broader``: parent id -> ordered child ids."""
        index: dict[str, list[str]] = {}
        for concept_id, concept in self.concepts.items():
            for parent_id in concept.broader:
                index.setdefault(parent_id, []).append(concept_id)
        return index

    def narrower(self, concept_id: str) -> list[str]:
        """Return the direct child (narrower) concept ids of ``concept_id``."""
        return self._narrower_index().get(concept_id, [])

    def children_of(self, concept_id: str) -> list[str]:
        """Alias of :meth:`narrower` that reads naturally at call sites."""
        return self.narrower(concept_id)

    @classmethod
    def from_graph(
        cls,
        g: "Graph",
        scheme_uri: "URIRef",
        concept_uris: "set[URIRef]",
    ) -> "ConceptScheme[Concept]":
        """Load from parsed graph."""
        concepts: dict[str, Concept] = dict(
            Concept.from_graph(g, cu) for cu in concept_uris
        )
        top = [cid for cid, c in concepts.items() if not c.broader]
        return ConceptScheme(
            uri=str(scheme_uri),
            title=get_scheme_title(g, scheme_uri),
            top_concepts=top,
            concepts=concepts,
        )

    def map_concepts(
        self, mapping_file: Path | None, attributes: list[Attribute]
    ) -> "MappedConceptScheme":
        """
        Map this scheme's concepts to attributes via a mapping file.

        The mapping file should consist of rows that conform to ConceptMappingRow,
        where each row links each attribute to its concept.
        """
        if mapping_file is None:
            # TODO: Implement mapping using external IDs in EppiJSON
            mapping_without_file = "Mapping without a mapping file is not yet supported"
            raise NotImplementedError(mapping_without_file)

        mapping = TypeAdapter(list[ConceptMappingRow]).validate_json(
            mapping_file.read_bytes()
        )
        mapping_dict = {row.attribute_name: row.concept_id for row in mapping}
        mapped_concepts: list[MappedConcept] = []
        for attribute in attributes:
            concept_id = mapping_dict.get(attribute.attribute_label)
            if concept_id is None:
                logger.warning(
                    f"Attribute {attribute.attribute_label} not in mapping file."
                )
                continue
            concept = self.get_concept(concept_id)
            if concept:
                mapped_concepts.append(
                    MappedConcept(**concept.model_dump(), attribute=attribute)
                )
            else:
                logger.warning(
                    f"Attribute {attribute.attribute_label}"
                    f" maps to concept {concept_id},"
                    " but concept is not contained in parsed vocabulary."
                )

        return MappedConceptScheme(
            title=self.title,
            description=self.description,
            uri=self.uri,
            top_concepts=[c.identifier for c in mapped_concepts if not c.broader],
            concepts={c.uri: c for c in mapped_concepts},
        )


class MappedConceptScheme(ConceptScheme[MappedConcept]):
    """A concept scheme where concepts have been mapped to attributes."""

    concepts: dict[str, MappedConcept] = Field(default_factory=dict)


def load_schemes_from_ttl(path: Path) -> list[ConceptScheme]:
    """Load a list of concept schemes from a ttl file."""
    g = parse_rdf(path.read_bytes(), detect_format(str(path)))
    result = analyze_graph(g)
    return [
        ConceptScheme.from_graph(g, s, result["concepts_by_scheme"].get(s, set()))
        for s in result["schemes"]
    ]
