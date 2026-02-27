"""Data models concerning `documents` and how to represent them in deet."""

import base64
import json
from collections.abc import Callable
from enum import StrEnum, auto
from io import BytesIO
from pathlib import Path
from random import randint
from typing import Self

from destiny_sdk.labs.references import LabsReference
from destiny_sdk.references import ReferenceFileInput
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from deet.data_models.base import GoldStandardAnnotation
from deet.exceptions import (
    BadDocumentIdError,
    MissingCitationElementError,
    NoAbstractError,
)
from deet.processors.parser import ParsedOutput
from deet.utils.identifier_utils import (
    DOCUMENT_ID_N_DIGITS,
    MAX_DOCUMENT_ID,
    MIN_DOCUMENT_ID,
    hash_n_strings_to_eight_digit_int,
)


class ContextType(StrEnum):
    """Types of context that can be provided to the LLM."""

    EMPTY = auto()
    FULL_DOCUMENT = auto()
    ABSTRACT_ONLY = auto()
    RAG_SNIPPETS = auto()
    CUSTOM = auto()


class DocumentIDSource(StrEnum):
    """
    Sources for a given document_id. Can be e.g. eppi_item_id.

    To be extended if e.g. we start working with
    non-eppi gold standard references.
    """

    EPPI_ITEM_ID = auto()
    DOI_AUTHOR_YEAR = auto()
    DOI_ID = auto()
    AUTHOR_YEAR_ID = auto()
    RANDINT = auto()


class DocumentIdentity(BaseModel):
    """A unified identity for a document, deriveable from multiple sources."""

    document_id: int | None = None
    document_id_source: DocumentIDSource | None = None

    # parsed citation info
    doi: str | None
    first_author: str | None
    year: str | None

    def populate_id(
        self,
        existing_ids: set[int] | None = None,
        hierarchy: list[DocumentIDSource] | None = None,
    ) -> None:
        """
        Populate document_id using a hierarchical list of ID creation methods.

        Tries each method in order until a unique ID is generated. If an ID
        conflicts with existing_ids, tries the next method. RANDINT always
        succeeds as fallback.

        NOTE: we will have to implement some sort of matching thing, if we are
        concerned that an id-collision might be becuase we have already
        parsed&linked a document.

        Args:
            existing_ids: List of existing IDs to check for conflicts.
            hierarchy: Ordered list of DocumentIDSource methods to try.
                Defaults to [EPPI_ITEM_ID, DOI_AUTHOR_YEAR, DOI_ID,
                AUTHOR_YEAR_ID, RANDINT].

        Raises:
            BadDocumentIdError: If unable to generate unique ID (should never
                happen as RANDINT is always in hierarchy).

        """
        if existing_ids is None:
            existing_ids = set()

        if hierarchy is None:
            hierarchy = list(DocumentIDSource)  # retain the order from the enum

        if DocumentIDSource.RANDINT not in hierarchy:
            hierarchy.append(
                DocumentIDSource.RANDINT
            )  # always keep random as a fallback

        attempted_sources = []

        for id_source in hierarchy:
            try:
                id_factory = self._create_id_factory(id_source)
                logger.debug(f"created id_factory: {id_factory}")
                potential_id = id_factory()
                logger.debug(f"created potential id: {potential_id}")

                # id collisions?
                if potential_id not in existing_ids:
                    self.document_id = potential_id
                    self.document_id_source = id_source
                    logger.debug(
                        f"successfully created document_id {potential_id} "
                        f"using {id_source}"
                    )
                    return

                logger.debug(
                    f"id {potential_id} from {id_source} conflicts with existing IDs"
                )
                attempted_sources.append(id_source)

            except (BadDocumentIdError, MissingCitationElementError) as e:
                logger.debug(f"Failed to create ID using {id_source}: {e}")
                attempted_sources.append(id_source)
                continue

        failed_sources = ", ".join(str(s) for s in attempted_sources)
        err_msg = (
            f"Failed to generate unique document_id after trying: {failed_sources}"
        )

        if len(attempted_sources) == len(hierarchy):
            max_attempts = 10
            attempts = 0
            for _ in range(max_attempts):
                potential_id = self._random_int_id()
                if potential_id not in existing_ids:
                    self.document_id = potential_id
                    self.document_id_source = DocumentIDSource.RANDINT
                    logger.debug(
                        f"successfully created document_id {potential_id} "
                        f"using {id_source}"
                    )
                    return

        err_msg += f" plus {attempts} randint attempts."
        raise BadDocumentIdError(err_msg)

    def _create_id_factory(self, id_source: DocumentIDSource) -> Callable:
        """
        Return an id-creating method given specific value of DocumentIDSource.

        Returns:
            int: the id.

        """
        id_creation_map = {
            DocumentIDSource.EPPI_ITEM_ID: self._eppi_item_id,
            DocumentIDSource.DOI_ID: self._doi_id,
            DocumentIDSource.AUTHOR_YEAR_ID: self._author_year_id,
            DocumentIDSource.DOI_AUTHOR_YEAR: self._doi_author_year_id,
            DocumentIDSource.RANDINT: self._random_int_id,
        }

        return id_creation_map[id_source]

    def _eppi_item_id(self) -> int:
        """Map an existing item_id (parsed as document_id)."""
        # we're going to assume that our `document_id`, received
        # from parsing eppi-json to EppiDocument is always going
        # to be eppi, otherwise this method should be extended to
        # reflect it coming from somewhere else.
        # either way, it'll have to be an 8-digit int.
        if (
            self.document_id is not None
            and isinstance(self.document_id, int)
            and len(str(self.document_id)) == DOCUMENT_ID_N_DIGITS
        ):
            return self.document_id
        bad_doc_id = f"id {self.document_id} is not a valid eppi item_id."
        raise BadDocumentIdError(bad_doc_id)

    def _citation_id_hasher(self, target_fields: list[str]) -> int:
        """Create an id from _n_ citation fields."""
        if not all(field in self.model_dump() for field in target_fields):
            missing_citation = (
                f"required fields are missing in citation. "
                "required: {', '.join(target_fields)}"
                f"actual: {','.join(self.model_dump())}"
            )
            raise MissingCitationElementError(missing_citation)
        payload = [self.model_dump()[field] for field in target_fields]

        if "" in payload or None in payload:
            none_or_empty = (
                "some or all of target fields are "
                f"None or empty strings: {','.join(target_fields)} "
            )
            raise MissingCitationElementError(none_or_empty)
        return hash_n_strings_to_eight_digit_int(payload)

    def _doi_id(self) -> int:
        """Create an integer id as a function of doi."""
        return self._citation_id_hasher(["doi"])

    def _doi_author_year_id(self) -> int:
        """Create an integer id as a function of doi, author and year."""
        return self._citation_id_hasher(["doi", "first_author", "year"])

    def _author_year_id(self) -> int:
        """Create an 8-digit integer id as a function of author and year."""
        return self._citation_id_hasher(["first_author", "year"])

    @staticmethod
    def _random_int_id() -> int:
        """Create a random integer id with 8 digits."""
        return randint(MIN_DOCUMENT_ID, MAX_DOCUMENT_ID)  # noqa: S311


class Document(BaseModel):
    """
    Represents a document.

    This can be used both for references itemised
    in a document listing gold standard annotations (e.g. eppi.json)
    AND
    for a document coming from a file (e.g. pdf) without
    linking to a gold standard annotations document with references.
    """

    # `extra` allows extra fields, e.g. for EppiDocument.
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    # `validate_assignment` runs model/field validators
    # not only on instantiation, but also when we change values,
    # e.g. when we set is_linked=True, thereby preventing
    # us from saying something is linked if it isn't.

    name: str
    citation: ReferenceFileInput
    context: str | None = None  # new defaults, empty
    context_type: ContextType | None = ContextType.EMPTY
    document_id: int | None = None
    document_identity: DocumentIdentity | None = None

    parsed_document: ParsedOutput | None = None
    original_doc_filepath: Path | None = (
        None  # NOTE -- add S3/blob support when required.
    )

    is_final: bool = False
    is_linked: bool = False

    @model_validator(mode="after")
    def validate_linking_complete(self) -> Self:
        """Validate linking is completed if `is_linked=True`."""
        base_err_msg = "requirements not met for linking: "
        if self.is_linked:
            if self.context is None:
                no_context_err = base_err_msg + "`context` is empty."
                raise ValueError(no_context_err)
            if self.document_identity is None:
                no_doc_id_err = (
                    base_err_msg + "`document_identity` is empty.  "
                    "run `init_document_identity() to populate."
                )
                raise ValueError(no_doc_id_err)
            if self.parsed_document is None:
                no_parsed_doc_err = base_err_msg + "`parsed_document` is empty. "
                raise ValueError(no_parsed_doc_err)

        return self

    @model_validator(mode="after")
    def validate_final(self) -> Self:
        """Validate Document is permitted to be `is_final`."""
        base_err_msg = "requirements not met for `Document().is_final`: "
        if self.is_final:
            if self.context is None:
                no_context_err = base_err_msg + "`context` is empty."
                raise ValueError(no_context_err)
            if self.document_identity is None:
                no_doc_id_err = (
                    base_err_msg + "`document_identity` is empty.  "
                    "run `init_document_identity() to populate."
                )
                raise ValueError(no_doc_id_err)

        return self

    def init_document_identity(
        self,
        existing_ids: set[int] | None = None,
        *,
        return_id: bool = True,
    ) -> int | None:
        """Initialise document_identity field using available metadata."""
        labs_ref = LabsReference(reference=self.citation)  # convert for easy access
        self.document_identity = DocumentIdentity(
            document_id=self.document_id,
            doi=labs_ref.doi,
            first_author=labs_ref.first_author,
            year=str(labs_ref.publication_year),
        )

        logger.info("populating id & id source...")
        self.document_identity.populate_id(existing_ids=existing_ids)
        if self.document_id is None:
            logger.info(
                "populating Document-level `document_id` field with "
                f"newly populated id {self.document_identity.document_id}... "
            )
            self.document_id = self.document_identity.document_id

        if return_id:
            return self.document_identity.document_id
        return None

    def author_year_from_document_identity(self) -> str:
        """
        Create lower-case `author_year` guess from a Document's
        DocumentIdentity field.
        The idea is to take the last name of the first author.

        NOTE: this can probably improved with more knowledge
        of how destiny encodes the first_author field.

        Returns:
            author_year (str): `author_year`

        """
        if self.document_identity is None:
            logger.debug("document identity is None, initialising...")
            self.init_document_identity()

        if (
            self.document_identity is None
            or self.document_identity.first_author is None
            or self.document_identity.year is None
        ):
            whats_what = f"self.document_identity: {self.document_identity}; "
            if self.document_identity is not None:
                whats_what += (
                    "self.document_identity.first_author: "
                    f"{self.document_identity.first_author}; "
                    f"self.document_identity.year: {self.document_identity.year}."
                )
            logger.warning(whats_what)
            raise ValueError
        author_name = self.document_identity.first_author
        year = self.document_identity.year

        name_components = author_name.split(" ")
        longest_component = max(name_components, key=len)

        return f"{longest_component.lower()}_{year}"

    def set_abstract_context(self) -> None:
        """Set the abstract, contained in `citation` field, as context."""
        abstract = LabsReference(reference=self.citation).abstract
        if abstract is not None:
            self.context_type = ContextType.ABSTRACT_ONLY
            self.context = abstract
            logger.info(
                "set context type to ABSTRACT_ONLY; set context to abstract."
                f" snippet: {abstract[:20]}"
            )
            return
        no_abstract = "No abstract found"
        raise NoAbstractError(no_abstract)

    def link_parsed_document(
        self,
        parsed_document: ParsedOutput,
        original_doc_filepath: Path | None = None,
    ) -> None:
        """
        Link parsed document and document metadata/`reference`.

        Args:
            parsed_document (ParsedOutput): the output from the parser
            original_doc_filepath (Path): full filepath to the original doc.

        """
        self.parsed_document = parsed_document
        if original_doc_filepath and (
            not original_doc_filepath.is_file() or not original_doc_filepath.exists()
        ):
            logger.warning(
                "supplied `original_doc_filepath` does not resolve. writing None."
            )
            original_doc_filepath = None
        self.original_doc_filepath = original_doc_filepath

    def set_context_from_parsed(self) -> Self:
        """Symlink context to parsed_document.text."""
        if self.parsed_document and self.parsed_document.text:
            self.context = self.parsed_document.text
            self.context_type = ContextType.FULL_DOCUMENT
        else:
            logger.warning("no text in parsed_document!")
        return self

    def save(self, path: Path) -> None:
        """Save linked document to .json."""
        data = self.model_dump(by_alias=False)

        # convert images to base64 for json serialization
        # NOTE @all -- we leave ourselves open to
        # malicious stuff being injected here. open to
        # suggestions as to how to validate/circumvent.
        # however, we do control what goes in and out,
        # so the danger here might be overstated.
        if self.parsed_document and self.parsed_document.images:
            images_b64 = {}
            for key, img in self.parsed_document.images.items():
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                images_b64[key] = base64.b64encode(buffer.getvalue()).decode()
            data["parsed_document"]["images"] = images_b64

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved LinkedDocument to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load linked document from .json."""
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug(data)
        if "citation" in data:
            logger.debug("we have a citationf ield")

        # convert base64 back to PIL img
        if data.get("parsed_document", {}).get("images"):
            images = {}
            for key, img_b64 in data["parsed_document"]["images"].items():
                img_bytes = base64.b64decode(img_b64)
                images[key] = Image.open(BytesIO(img_bytes))
            data["parsed_document"]["images"] = images

        return cls(**data)


class GoldStandardAnnotatedDocument(Document):
    """A document with its gold standard annotations."""

    annotations: list[GoldStandardAnnotation]
