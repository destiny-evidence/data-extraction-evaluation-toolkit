"""Utilities for working with destiny data types."""

from destiny_sdk.enhancements import (
    AuthorPosition,
    Authorship,
    BibliographicMetadataEnhancement,
    EnhancementType,
    LocationEnhancement,
)
from destiny_sdk.references import Reference, ReferenceFileInput
from ftfy import fix_text


class ReferencePresenter:
    """A class to present references in a Streamlit application."""

    def __init__(self, reference: Reference | ReferenceFileInput) -> None:
        """Init the ReferencePresenter class."""
        self.reference = reference

    @property
    def bibliographic_metadata(self) -> BibliographicMetadataEnhancement | None:
        """Returns the bibliographic metadata enhancement of the reference."""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.BIBLIOGRAPHIC:
                return e.content
        return None

    @property
    def location(self) -> LocationEnhancement | None:
        """Returns the location enhancement of the reference."""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.LOCATION:
                return e.content
        return None

    @property
    def title(self) -> str:
        """Returns the title of the reference."""
        return (
            fix_text(self.bibliographic_metadata.title) or ""
            if self.bibliographic_metadata
            else ""
        )

    @property
    def authors(self) -> str:
        """Returns the authors of the reference as a semi-colon-separated string."""
        if self.bibliographic_metadata and self.bibliographic_metadata.authorship:
            return "; ".join(
                [a.display_name for a in self.bibliographic_metadata.authorship]
            )
        return ""

    @property
    def orcids(self) -> str:
        """
        Returns all available ORCIDs for each author of
        the reference as a semi-colon-separated string.
        """
        if self.bibliographic_metadata and self.bibliographic_metadata.authorship:
            return "; ".join(
                [str(a.orcid) for a in self.bibliographic_metadata.authorship]
            )
        return ""

    @property
    def first_author(self) -> Authorship | None:
        """Returns the first author of the reference."""
        if self.bibliographic_metadata and self.bibliographic_metadata.authorship:
            for author in self.bibliographic_metadata.authorship:
                if author.position == AuthorPosition.FIRST:
                    return author
        return None

    @property
    def publisher(self) -> str:
        """Returns the publisher of the reference."""
        return (
            self.bibliographic_metadata.publisher or ""
            if self.bibliographic_metadata
            else ""
        )

    @property
    def publication_date(self) -> str:
        """Returns the publication date of the reference in 'dd mmm yyyy' format."""
        if self.bibliographic_metadata:
            pub_date = self.bibliographic_metadata.publication_date
            if pub_date:
                return pub_date.strftime("%d %b %Y") or ""
        return ""

    @property
    def year(self) -> str:
        """Returns the publication year of the reference."""
        return (
            str(
                self.bibliographic_metadata.publication_year
                or (
                    self.bibliographic_metadata.publication_date.year
                    if self.bibliographic_metadata.publication_date
                    else ""
                )
            )
            if self.bibliographic_metadata
            else ""
        )

    @property
    def publication_locations(self) -> str:
        """
        Returns the publication locations of the
        reference as a semi-colon-separated string.
        """
        if self.location:
            return (
                "; ".join(
                    [
                        str(loc.extra["display_name"]) or ""
                        for loc in self.location.locations
                        if loc.extra
                    ]
                )
                or ""
                if self.location.locations
                else ""
            )
        return ""

    @property
    def publication_links(self) -> str:
        """
        Returns the publication location links of the
        reference as a semi-colon-separated string.
        """
        if self.location:
            return (
                "; ".join(
                    [str(loc.landing_page_url) or "" for loc in self.location.locations]
                )
                or ""
                if self.location.locations
                else ""
            )
        return ""

    @property
    def publication_types(self) -> str:
        """
        Returns the publication types of the
        reference as a semi-colon-separated string.
        """
        if self.location:
            return (
                "; ".join(
                    [
                        str(loc.extra["type"]) or ""
                        for loc in self.location.locations
                        if loc.extra
                    ]
                )
                or ""
                if self.location.locations
                else ""
            )
        return ""

    @property
    def doi(self) -> str | None:
        """Returns the DOI of the reference."""
        for i in self.reference.identifiers or []:
            if i.identifier_type == "doi":
                return f"https://doi.org/{i.identifier}"
        return None

    @property
    def abstract(self) -> str:
        """Returns the abstract of the reference."""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ABSTRACT:
                return fix_text(e.content.abstract)
        return ""

    @property
    def topics(self) -> str:
        """
        Returns the topic labels of the reference and their id links
        as a semi-colon-separated string of tuples.
        """
        a = ""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ANNOTATION:
                for x in e.content.annotations or []:
                    if x.scheme == "openalex:topic":
                        a += str(((x.label.title()), x.data["id"] if x.data else ""))
                        a += ";"
        return a.strip(";")

    @property
    def topic_domains(self) -> str:
        """
        Returns the topic domain labels of the reference and their id links
        as a semi-colon-separated string of tuples.
        """
        a = ""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ANNOTATION:
                for x in e.content.annotations or []:
                    if x.scheme == "openalex:topic" and x.data:
                        a += str(
                            (
                                str(x.data["domain"]["display_name"]).title(),
                                x.data["domain"]["id"],
                            )
                        )
                        a += ";"
        return a.strip(";")

    @property
    def topic_fields(self) -> str:
        """
        Returns the topic field labels of the reference and their id links
        as a semi-colon-separated string of tuples.
        """
        a = ""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ANNOTATION:
                for x in e.content.annotations or []:
                    if x.scheme == "openalex:topic" and x.data:
                        a += str(
                            (
                                str(x.data["field"]["display_name"]).title(),
                                x.data["field"]["id"],
                            )
                        )
                        a += ";"
        return a.strip(";")

    @property
    def topic_sub_fields(self) -> str:
        """
        Returns the topic sub-field labels of the reference and their id links
        as a semi-colon-separated string of tuples.
        """
        a = ""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ANNOTATION:
                for x in e.content.annotations or []:
                    if x.scheme == "openalex:topic" and x.data:
                        a += str(
                            (
                                str(x.data["subfield"]["display_name"]).title(),
                                x.data["subfield"]["id"],
                            )
                        )
                        a += ";"
        return a.strip(";")

    @property
    def taxonomy(self) -> str:
        """
        Returns the taxonomy labels of the reference as
        a semi-colon-separated string of tuples.
        """
        a = ""
        for e in self.reference.enhancements or []:
            if e.content.enhancement_type == EnhancementType.ANNOTATION:
                for x in e.content.annotations or []:
                    if x.scheme.startswith("classifier:taxonomy") and x.value:
                        a += str((x.scheme.split(":")[-1], x.label)).title()
                        a += ";"
        return a.strip(";")

    def to_dict(self) -> dict:
        """
        Convert the reference to a dictionary suitable
        for DataFrame representation.
        """
        return {
            "title": self.title,
            "authors": self.authors,
            "orcids": self.orcids,
            "publisher": self.publisher,
            "publication_date": self.publication_date,
            "year": self.year,
            "publication_locations": self.publication_locations,
            "publication_links": self.publication_links,
            "publication_types": self.publication_types,
            "doi": self.doi,
            "abstract": self.abstract,
            "topics": self.topics,
            "topic_domains": self.topic_domains,
            "topic_fields": self.topic_fields,
            "topic_sub_fields": self.topic_sub_fields,
            "taxonomy": self.taxonomy,
        }
