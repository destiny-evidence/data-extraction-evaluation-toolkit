# Vendored from https://github.com/destiny-evidence/taxonomy-builder/blob/ce0e833048ad4a0a430f520571b3b4e854d0f523/backend/src/taxonomy_builder/services/rdf_parser.py
"""Pure RDF parsing and analysis — no database access."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal as LiteralType
from urllib.parse import urlparse

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

class RestrictionType(StrEnum):
    """Type of OWL restriction recorded on an ontology class."""

    ALL_VALUES_FROM = "allValuesFrom"
    SOME_VALUES_FROM = "someValuesFrom"
    HAS_VALUE = "hasValue"

WELL_KNOWN_SUPERCLASS_URIREFS: frozenset[URIRef] = frozenset({
    SKOS.Concept,
    OWL.Thing,
    RDFS.Resource,
})

# XSD namespace prefix for abbreviating datatype URIs
XSD_NS = str(XSD)

# File extension to RDFLib format mapping
FORMAT_MAP = {
    ".ttl": "turtle",
    ".turtle": "turtle",
    ".rdf": "xml",
    ".xml": "xml",
    ".owl": "xml",
    ".jsonld": "json-ld",
    ".json": "json-ld",
    ".nt": "nt",
    ".n3": "n3",
}


class InvalidRDFError(Exception):
    """RDF file could not be parsed."""

    def __init__(self, message: str = "Could not parse RDF file") -> None:
        super().__init__(message)


def detect_format(filename: str) -> str:
    """Detect RDF format from filename extension."""
    filename_lower = filename.lower()
    for ext, fmt in FORMAT_MAP.items():
        if filename_lower.endswith(ext):
            return fmt
    raise InvalidRDFError(
        f"Unsupported file format. Supported formats: {', '.join(FORMAT_MAP.keys())}"
    )


def parse_rdf(content: bytes, fmt: str) -> Graph:
    """Parse RDF content into a graph."""
    # "none" ensures only prefixes declared in the file are bound,
    # not rdflib's 29 built-in defaults (brick, csvw, doap, …).
    g = Graph(bind_namespaces="none")
    try:
        g.parse(data=content, format=fmt)
    except Exception as e:
        raise InvalidRDFError(f"Failed to parse RDF: {e}") from e
    return g


# --- Validation ---


@dataclass
class ValidationIssue:
    severity: LiteralType["error", "warning", "info"]
    type: str
    message: str
    entity_uri: str | None = None


@dataclass
class ValidationResult:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


def validate_graph(g: Graph, class_uris: set[str]) -> ValidationResult:
    """Run validation checks on a parsed RDF graph before import.

    Args:
        g: The parsed RDF graph.
        class_uris: URIs of known classes (existing project classes + classes in the file).

    Returns:
        ValidationResult with categorised issues.
    """
    result = ValidationResult()

    if len(g) == 0:
        result.warnings.append(
            ValidationIssue(
                severity="warning",
                type="empty_graph",
                message=(
                    "File parsed successfully but contains no RDF data — "
                    "check the file format matches the file extension"
                ),
            )
        )
        return result

    _check_uri_schemes(g, result)
    _check_unresolved_domains(g, class_uris, result)
    _check_rdf_properties(g, result)
    _check_superclass_cycles(g, result)
    _check_unsupported_named_individuals(g, result)
    _check_unsupported_restrictions(g, result)

    return result


_ALLOWED_URI_SCHEMES = {"http", "https"}


def _check_uri_schemes(g: Graph, result: ValidationResult) -> None:
    """Detect URIs with non-http(s) schemes in subjects and objects."""
    seen: set[str] = set()
    for s, _p, o in g:
        for node in (s, o):
            if isinstance(node, URIRef):
                uri_str = str(node)
                if uri_str in seen:
                    continue
                scheme = urlparse(uri_str).scheme
                if scheme in _ALLOWED_URI_SCHEMES:
                    continue
                seen.add(uri_str)
                if scheme == "file":
                    result.errors.append(
                        ValidationIssue(
                            severity="error",
                            type="file_uri",
                            message=(
                                f"file:// URI detected: {uri_str} — "
                                f"this usually means the file is missing an @base directive"
                            ),
                            entity_uri=uri_str,
                        )
                    )
                else:
                    result.errors.append(
                        ValidationIssue(
                            severity="error",
                            type="unsupported_uri_scheme",
                            message=(
                                f"Unsupported URI scheme '{scheme}:' in {uri_str} — "
                                f"only http:// and https:// URIs are supported"
                            ),
                            entity_uri=uri_str,
                        )
                    )


def _check_unresolved_domains(g: Graph, class_uris: set[str], result: ValidationResult) -> None:
    """Warn when properties reference domain classes not in the known set."""
    # Collect classes defined in this graph
    file_class_uris = {str(uri) for uri in find_owl_classes(g)}
    all_class_uris = class_uris | file_class_uris

    for uri, _ptype in find_properties(g):
        domain = g.value(uri, RDFS.domain)
        # Collect domain URIs: plain URIRef or union members
        domain_strs: list[str] = []
        if isinstance(domain, URIRef):
            domain_strs = [str(domain)]
        elif isinstance(domain, BNode):
            domain_strs = _resolve_union_all(g, domain)

        for domain_str in domain_strs:
            if domain_str not in all_class_uris:
                result.warnings.append(
                    ValidationIssue(
                        severity="warning",
                        type="unresolved_domain",
                        message=(
                            f"Property '{get_identifier_from_uri(uri)}' has domain "
                            f"'{domain_str}' which doesn't match any class "
                            f"in the project"
                        ),
                        entity_uri=str(uri),
                    )
                )


def _check_rdf_properties(g: Graph, result: ValidationResult) -> None:
    """Detect rdf:Property instances for informational reporting."""
    for subject in g.subjects(RDF.type, RDF.Property):
        if isinstance(subject, URIRef):
            result.info.append(
                ValidationIssue(
                    severity="info",
                    type="rdf_property",
                    message=(
                        f"'{get_identifier_from_uri(subject)}' is typed as rdf:Property "
                        f"(not owl:ObjectProperty/DatatypeProperty)"
                    ),
                    entity_uri=str(subject),
                )
            )


def _check_superclass_cycles(g: Graph, result: ValidationResult) -> None:
    """Detect cycles in rdfs:subClassOf edges between ontology classes."""
    # Exclude only well-known external URIs from cycle analysis, not all
    # concept-typed classes. Since #144, concept-typed classes are part of the
    # ontology and their edges must participate in cycle detection.
    owl_classes = find_owl_classes(g)
    class_metadata = [
        extract_class_metadata(
            g,
            cls,
            exclude_superclass_uris=WELL_KNOWN_SUPERCLASS_URIREFS,
        )
        for cls in owl_classes
    ]
    cycles = detect_superclass_cycles(class_metadata)
    if cycles:
        cycle_desc = ", ".join(f"{a} → {b}" for a, b in cycles)
        result.errors.append(
            ValidationIssue(
                severity="error",
                type="superclass_cycle",
                message=f"Cycle detected in rdfs:subClassOf hierarchy: {cycle_desc}",
            )
        )


def find_named_individuals(g: Graph) -> list[tuple[str, str]]:
    """Extract owl:NamedIndividual instances as (uri, label) pairs."""
    individuals: list[tuple[str, str]] = []
    for subject in g.subjects(RDF.type, OWL.NamedIndividual):
        if isinstance(subject, URIRef):
            uri = str(subject)
            label = g.value(subject, RDFS.label)
            individuals.append((uri, str(label) if label else uri.rsplit("/", 1)[-1]))
    return sorted(individuals)


def _check_unsupported_named_individuals(g: Graph, result: ValidationResult) -> None:
    """Detect owl:NamedIndividual instances."""
    count = sum(1 for s in g.subjects(RDF.type, OWL.NamedIndividual) if isinstance(s, URIRef))
    if count:
        result.info.append(
            ValidationIssue(
                severity="info",
                type="unsupported_named_individual",
                message=(
                    f"Found {count} named individual{'s' if count != 1 else ''} "
                    f"— not yet supported (#112)"
                ),
            )
        )


def _check_unsupported_restrictions(g: Graph, result: ValidationResult) -> None:
    """Detect OWL restrictions we don't handle and warn about them.

    Value restrictions (allValuesFrom, someValuesFrom, hasValue) are handled
    and stored in the class_restriction table. Only warn about cardinality
    restrictions and unrecognised types.
    """
    handled = {OWL.allValuesFrom, OWL.someValuesFrom, OWL.hasValue}
    restrictions: list[str] = []
    for subject in g.subjects(RDF.type, OWL.Restriction):
        on_prop = g.value(subject, OWL.onProperty)
        prop_label = get_identifier_from_uri(on_prop) if isinstance(on_prop, URIRef) else "?"

        # Skip value restrictions — we handle them
        if any(g.value(subject, pred) is not None for pred in handled):
            continue

        restrictions.append(f"{prop_label} (unrecognised restriction type)")

    if restrictions:
        details = "; ".join(restrictions)
        result.warnings.append(
            ValidationIssue(
                severity="warning",
                type="unsupported_restriction",
                message=(
                    f"Found {len(restrictions)} unsupported OWL "
                    f"restriction{'s' if len(restrictions) != 1 else ''} "
                    f"that will be dropped on import: {details}"
                ),
            )
        )


# --- URI helpers ---


def get_identifier_from_uri(uri: URIRef) -> str:
    """Extract identifier (local name) from URI."""
    uri_str = str(uri)
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.rstrip("/").split("/")[-1]


def is_xsd_type(uri_str: str) -> bool:
    """Return True if the URI is in the XSD namespace."""
    return uri_str.startswith(XSD_NS)


def abbreviate_xsd(uri_str: str) -> str:
    """Convert full XSD URI to xsd: prefix form, or return as-is."""
    if is_xsd_type(uri_str):
        return "xsd:" + uri_str[len(XSD_NS) :]
    return uri_str


# --- SKOS concept/scheme helpers ---


def find_all_concepts(g: Graph) -> set[URIRef]:
    """Find all concepts including those typed as subclasses of skos:Concept."""
    concepts: set[URIRef] = set()

    for instance in g.subjects(RDF.type, SKOS.Concept):
        if isinstance(instance, URIRef):
            concepts.add(instance)

    for concept_class in g.transitive_subjects(RDFS.subClassOf, SKOS.Concept):
        for instance in g.subjects(RDF.type, concept_class):
            if isinstance(instance, URIRef):
                concepts.add(instance)

    return concepts


def extract_concept_type_uris(g: Graph, concept_uri: URIRef) -> list[str]:
    """Collect non-skos:Concept rdf:type URIs for a concept. Sorted and deduplicated.

    Captures ALL rdf:type URIs except skos:Concept itself, for round-trip fidelity.
    This includes concept-typed classes (subClassOf skos:Concept), utility marker
    classes, owl:NamedIndividual, etc.
    """
    type_uris: set[str] = set()
    for rdf_type in g.objects(concept_uri, RDF.type):
        if not isinstance(rdf_type, URIRef):
            continue
        if rdf_type == SKOS.Concept:
            continue
        type_uris.add(str(rdf_type))
    return sorted(type_uris)


def find_concept_subclasses(g: Graph) -> set[URIRef]:
    """Find owl:Class URIs that are rdfs:subClassOf skos:Concept."""
    subclasses: set[URIRef] = set()
    for cls in g.transitive_subjects(RDFS.subClassOf, SKOS.Concept):
        if isinstance(cls, URIRef) and cls != SKOS.Concept:
            subclasses.add(cls)
    return subclasses


def get_scheme_title(g: Graph, scheme_uri: URIRef) -> str:
    """Get scheme title with priority: rdfs:label > skos:prefLabel > dcterms:title > URI."""
    label = g.value(scheme_uri, RDFS.label)
    if label:
        return str(label)

    label = g.value(scheme_uri, SKOS.prefLabel)
    if label:
        return str(label)

    label = g.value(scheme_uri, DCTERMS.title)
    if label:
        return str(label)

    return get_identifier_from_uri(scheme_uri)


def get_scheme_description(g: Graph, scheme_uri: URIRef) -> str | None:
    """Get scheme description from rdfs:comment or dcterms:description."""
    desc = g.value(scheme_uri, RDFS.comment)
    if desc:
        return str(desc)
    desc = g.value(scheme_uri, DCTERMS.description)
    if desc:
        return str(desc)
    return None


def get_concept_scheme(g: Graph, concept_uri: URIRef) -> URIRef | None:
    """Get the scheme a concept belongs to via skos:inScheme or skos:topConceptOf."""
    scheme = g.value(concept_uri, SKOS.inScheme)
    if scheme and isinstance(scheme, URIRef):
        return scheme

    scheme = g.value(concept_uri, SKOS.topConceptOf)
    if scheme and isinstance(scheme, URIRef):
        return scheme

    return None


def get_concept_pref_label(g: Graph, concept_uri: URIRef) -> tuple[str, str | None]:
    """Get prefLabel for concept, returning (label, warning) tuple."""
    label = g.value(concept_uri, SKOS.prefLabel)
    if label:
        return str(label), None

    local_name = get_identifier_from_uri(concept_uri)
    warning = f"Concept {concept_uri} has no prefLabel, using URI fragment: {local_name}"
    return local_name, warning


def count_broader_relationships(g: Graph, concepts: set[URIRef]) -> int:
    """Count broader relationships among the given concepts."""
    count = 0
    for concept in concepts:
        for broader in g.objects(concept, SKOS.broader):
            if isinstance(broader, URIRef) and broader in concepts:
                count += 1
    return count


# --- OWL class helpers ---


def find_owl_classes(g: Graph) -> list[URIRef]:
    """Find owl:Class instances, including concept-typed classes (subClassOf skos:Concept)."""
    classes: list[URIRef] = []
    for subject in g.subjects(RDF.type, OWL.Class):
        if not isinstance(subject, URIRef):
            continue
        if (subject, OWL.unionOf, None) in g:
            continue
        classes.append(subject)

    return classes


def extract_class_metadata(
    g: Graph,
    class_uri: URIRef,
    *,
    exclude_superclass_uris: set[URIRef] | None = None,
) -> dict:
    """Extract label, description, scope_note, and superclass_uris for an OWL class."""
    uri_str = str(class_uri)
    if exclude_superclass_uris is None:
        exclude_superclass_uris = set()

    label = g.value(class_uri, RDFS.label)
    if not label:
        label = get_identifier_from_uri(class_uri)
    else:
        label = str(label)

    description = g.value(class_uri, RDFS.comment)
    scope_note = g.value(class_uri, SKOS.scopeNote)

    # Collect superclass URIs, filtering out blank nodes and excluded URIs
    superclass_uris: list[str] = []
    for obj in g.objects(class_uri, RDFS.subClassOf):
        if not isinstance(obj, URIRef):
            continue
        if obj in exclude_superclass_uris:
            continue
        superclass_uris.append(str(obj))

    return {
        "identifier": get_identifier_from_uri(class_uri),
        "label": str(label),
        "description": str(description) if description else None,
        "scope_note": str(scope_note) if scope_note else None,
        "uri": uri_str,
        "superclass_uris": sorted(superclass_uris),
    }


def extract_restrictions(g: Graph, class_uris: set[str]) -> list[dict]:
    """Extract value restrictions from rdfs:subClassOf blank nodes.

    Handles allValuesFrom, someValuesFrom, and hasValue.
    Returns list of dicts: {class_uri, on_property_uri, restriction_type, value_uri}.
    """
    _OWL_PREDICATES = [
        (OWL.allValuesFrom, RestrictionType.ALL_VALUES_FROM),
        (OWL.someValuesFrom, RestrictionType.SOME_VALUES_FROM),
        (OWL.hasValue, RestrictionType.HAS_VALUE),
    ]

    restrictions: list[dict] = []
    for class_uri_str in sorted(class_uris):
        class_uri = URIRef(class_uri_str)
        for obj in g.objects(class_uri, RDFS.subClassOf):
            if not isinstance(obj, BNode):
                continue
            if (obj, RDF.type, OWL.Restriction) not in g:
                continue
            on_prop = g.value(obj, OWL.onProperty)
            if not isinstance(on_prop, URIRef):
                continue
            for predicate, rtype in _OWL_PREDICATES:
                value = g.value(obj, predicate)
                if value is None:
                    continue
                if not isinstance(value, URIRef):
                    # Skip anonymous class expressions (blank-node fillers)
                    continue
                restrictions.append(
                    {
                        "class_uri": class_uri_str,
                        "on_property_uri": str(on_prop),
                        "restriction_type": rtype.value,
                        "value_uri": str(value),
                    }
                )
                break  # one predicate per restriction bnode
    return restrictions


def detect_superclass_cycles(classes_metadata: list[dict]) -> list[tuple[str, str]]:
    """Detect cycles in superclass edges via DFS. Returns list of cycle-forming edges."""
    # Build adjacency: child → superclasses
    superclasses_by_uri: dict[str, list[str]] = {}
    for cm in classes_metadata:
        superclasses_by_uri[cm["uri"]] = cm.get("superclass_uris", [])

    UNVISITED, IN_PROGRESS, COMPLETE = 0, 1, 2
    status: dict[str, int] = {uri: UNVISITED for uri in superclasses_by_uri}
    cycle_edges: list[tuple[str, str]] = []

    def visit(node: str) -> None:
        status[node] = IN_PROGRESS
        for parent in superclasses_by_uri.get(node, []):
            if parent not in status:
                continue  # external URI, skip
            if status[parent] == IN_PROGRESS:
                cycle_edges.append((node, parent))
            elif status[parent] == UNVISITED:
                visit(parent)
        status[node] = COMPLETE

    for uri in superclasses_by_uri:
        if status[uri] == UNVISITED:
            visit(uri)

    return cycle_edges


# --- OWL property helpers ---


def find_properties(g: Graph) -> list[tuple[URIRef, str]]:
    """Find owl:ObjectProperty, owl:DatatypeProperty, and rdf:Property instances.

    Returns list of (uri, property_type) tuples, deduplicated by URI.
    If a property is typed as both ObjectProperty and DatatypeProperty,
    the type is resolved from rdfs:range: XSD range -> datatype,
    otherwise -> object.  rdf:Property instances get type "rdf".
    """
    object_props: set[URIRef] = set()
    datatype_props: set[URIRef] = set()
    rdf_props: set[URIRef] = set()

    for subject in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(subject, URIRef):
            object_props.add(subject)

    for subject in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(subject, URIRef):
            datatype_props.add(subject)

    for subject in g.subjects(RDF.type, RDF.Property):
        if isinstance(subject, URIRef):
            rdf_props.add(subject)

    owl_uris = object_props | datatype_props
    properties: list[tuple[URIRef, str]] = []

    for uri in owl_uris:
        is_obj = uri in object_props
        is_dt = uri in datatype_props
        if is_obj and is_dt:
            range_val = g.value(uri, RDFS.range)
            if range_val and str(range_val).startswith(str(XSD)):
                properties.append((uri, "datatype"))
            else:
                properties.append((uri, "object"))
        elif is_dt:
            properties.append((uri, "datatype"))
        else:
            properties.append((uri, "object"))

    # Add rdf:Property instances not already covered by OWL types
    for uri in rdf_props - owl_uris:
        properties.append((uri, "rdf"))

    return properties


def _resolve_union_all(g: Graph, bnode: BNode) -> list[str]:
    """Extract all URIRefs from an owl:unionOf RDF Collection on a blank node."""
    union_list = g.value(bnode, OWL.unionOf)
    if union_list is None:
        return []
    uris: list[str] = []
    seen_nodes: set = set()
    seen_uris: set[str] = set()
    node = union_list
    while node is not None and node != RDF.nil:
        if node in seen_nodes:
            break  # circular rdf:rest — bail out
        seen_nodes.add(node)
        first = g.value(node, RDF.first)
        if isinstance(first, URIRef):
            uri = str(first)
            if uri not in seen_uris:
                uris.append(uri)
                seen_uris.add(uri)
        node = g.value(node, RDF.rest)
    return uris


def extract_property_metadata(g: Graph, prop_uri: URIRef, prop_type: str) -> dict:
    """Extract metadata for a property."""
    label = g.value(prop_uri, RDFS.label)
    if not label:
        label = get_identifier_from_uri(prop_uri)
    else:
        label = str(label)

    description = g.value(prop_uri, RDFS.comment)
    domain = g.value(prop_uri, RDFS.domain)
    range_val = g.value(prop_uri, RDFS.range)

    # Resolve domain: plain URIRef → [uri], BNode with owl:unionOf → all members
    domain_uris: list[str] = []
    if isinstance(domain, URIRef):
        domain_uris = [str(domain)]
    elif isinstance(domain, BNode):
        domain_uris = _resolve_union_all(g, domain)

    # Read allowMultiple annotation for cardinality (any namespace)
    cardinality = "single"
    for pred, obj in g.predicate_objects(prop_uri):
        if str(pred).endswith("allowMultiple") and isinstance(obj, Literal):
            if obj.toPython() is True:
                cardinality = "multiple"
            break

    return {
        "identifier": get_identifier_from_uri(prop_uri),
        "label": str(label),
        "description": str(description) if description else None,
        "property_type": prop_type,
        "domain_uris": domain_uris,
        "range_uri": str(range_val) if isinstance(range_val, URIRef) else None,
        "uri": str(prop_uri),
        "cardinality": cardinality,
    }


def resolve_object_range(
    g: Graph,
    range_uri: str,
    scheme_uris: set[str],
    class_uris: set[str],
) -> tuple[str, str] | None:
    """Resolve an object property range URI.

    Returns ("scheme", scheme_uri), ("class", range_uri),
    ("ambiguous", range_uri), or None.

    Strategies:
    1. RDF linkage: find concepts typed with the range class, follow inScheme
    2. Direct scheme URI match
    3. Class URI match
    """
    range_ref = URIRef(range_uri)

    # Strategy 1: Follow RDF linkage to a scheme
    matched_schemes: set[str] = set()
    for instance in g.subjects(RDF.type, range_ref):
        if not isinstance(instance, URIRef):
            continue
        scheme = get_concept_scheme(g, instance)
        if scheme and str(scheme) in scheme_uris:
            matched_schemes.add(str(scheme))
    if len(matched_schemes) == 1:
        return ("scheme", next(iter(matched_schemes)))
    if len(matched_schemes) > 1:
        return ("ambiguous", range_uri)

    # Strategy 2: Direct scheme URI match
    if range_uri in scheme_uris:
        return ("scheme", range_uri)

    # Strategy 3: Class URI match
    if range_uri in class_uris:
        return ("class", range_uri)

    return None


# --- Graph analysis ---


def analyze_graph(g: Graph) -> dict:
    """Analyze the RDF graph and extract all entity types."""
    schemes: list[URIRef] = []
    for scheme in g.subjects(RDF.type, SKOS.ConceptScheme):
        if isinstance(scheme, URIRef):
            schemes.append(scheme)

    all_concepts = find_all_concepts(g)

    concepts_by_scheme: dict[URIRef, set[URIRef]] = {s: set() for s in schemes}
    orphan_concepts: set[URIRef] = set()

    for concept in all_concepts:
        scheme = get_concept_scheme(g, concept)
        if scheme and scheme in concepts_by_scheme:
            concepts_by_scheme[scheme].add(concept)
        else:
            orphan_concepts.add(concept)

    warnings: list[str] = []
    if orphan_concepts:
        if len(schemes) == 1:
            concepts_by_scheme[schemes[0]].update(orphan_concepts)
        else:
            for orphan in orphan_concepts:
                warnings.append(f"Concept {orphan} has no scheme membership and was skipped")

    owl_classes = find_owl_classes(g)
    class_metadata = [extract_class_metadata(g, cls) for cls in owl_classes]

    class_uri_set = {cm["uri"] for cm in class_metadata}
    restrictions = extract_restrictions(g, class_uri_set)

    owl_properties = find_properties(g)
    property_metadata = [extract_property_metadata(g, uri, ptype) for uri, ptype in owl_properties]

    return {
        "schemes": schemes,
        "concepts_by_scheme": concepts_by_scheme,
        "warnings": warnings,
        "classes": class_metadata,
        "properties": property_metadata,
        "restrictions": restrictions,
    }
