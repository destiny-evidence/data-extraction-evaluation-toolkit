"""Tools for linking references/citations with parsed documents."""

import csv
import json
from collections.abc import Callable, Generator
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

from deet.data_models.documents import Document
from deet.exceptions import JsonStyleError
from deet.processors.parser import DocumentParser, ParsedOutput
from deet.utils.identifier_utils import DOCUMENT_ID_N_DIGITS

parser = DocumentParser()


class LinkingStrategy(StrEnum):
    """Enum of permitted/implemented ref<>parsed_doc linking strategies."""

    MAPPING_FILE = auto()
    FILENAME_AUTHOR_YEAR_LONGEST = auto()
    FILENAME_AUTHOR_YEAR_LAST = auto()
    FILENAME_ID = auto()
    # to add: some sort of interactive selection thing in CLI
    # or, later on, in UI.


class DocumentReferenceMapping(BaseModel):
    """
    Data model for incoming, manual mappings
    of references (via 8-digit integer ids) to
    documents, via filename.
    """

    document_id: int
    file_path: Path
    # NOTE: may want to inherit this from supported
    # formats in parser.py
    format: Literal["md", "pdf"] | None = None

    @field_validator("document_id", mode="before")
    @classmethod
    def ensure_valid_doc_id(cls, value: int) -> int:
        """Ensure supplied document_id has 8 digits."""
        if len(str(value)) != DOCUMENT_ID_N_DIGITS:
            val_err = f'`document_id` must have 8 digits. supplied" {value}'
            raise ValueError(val_err)
        return value

    @model_validator(mode="after")
    def ensure_file_exists(self) -> Self:
        """
        Ensure either md_path or pdf_path are populated,
        and the associated file exists.
        """
        # check path resolves
        if not self.file_path.is_file():
            not_a_file = f"{self.file_path} is not a file."
            raise ValueError(not_a_file)
        # check suffix
        if self.file_path.suffix not in [".md", ".pdf"]:
            unsupported = (
                f"{self.file_path.suffix} is not supported. either .md or .pdf"
            )
            raise ValueError(unsupported)
        self.format = self.file_path.suffix[1:]  # type:ignore[assignment] # omit .

        return self


class LinkedInterimPayload(DocumentReferenceMapping):
    """
    Interim output from a linking factory method; extending
    DocumentReferenceMapping.


    Interim as the document may
    a) still need to be parsed, and
    b) still needs to be coerced into ParsedOutput.
    """

    unlinked_document: Document


class MappingImporter:
    """
    Tool for importing manual mappings from csv/json
    to list[DocumentReferenceMapping].
    """

    def __init__(
        self, mapping_file_path: Path, document_base_dir: Path | None = None
    ) -> None:
        """
        Initialise MappingImporter instance.


        Args:
            mapping_file_path (Path): Path to csv/json file containing mappings.
            document_base_dir (Path | None, optional): Optional directory path
            where documents (pdf/md) live. Defaults to None.


        """
        if mapping_file_path.suffix not in [".csv", ".json"]:
            bad_file = "mapping file needs to be supplied in csv or json format."
            raise ValueError(bad_file)

        if document_base_dir and not document_base_dir.is_dir():
            bad_dir = "document_base_dir is optional, but must be a dir if supplied."
            raise ValueError(bad_dir)

        self.mapping_file_path = mapping_file_path
        self.mapping_file_type = mapping_file_path.suffix[1:]
        self.document_base_dir = document_base_dir

    def import_mapping(self) -> list[DocumentReferenceMapping]:
        """Parse a csv/json file to a list od DocumentReferenceMapping objects."""
        if self.mapping_file_type == "json":
            payload = self._load_json()
        elif self.mapping_file_type == "csv":
            payload = self._load_csv()
        else:
            bad_file = "only json or csv files are supported."
            raise ValueError(bad_file)

        logger.info(
            f"supplied mapping file {self.mapping_file_path.name} "
            f"is {self.mapping_file_type}."
        )

        return [
            DocumentReferenceMapping(document_id=doc_id, file_path=file_path)
            for doc_id, file_path in payload.items()
        ]

    def _load_json(self) -> dict[int, Path]:
        """
        Load json file with mappings to dict.


        NOTE: json can be in...
        - array style:
        [
        {"document_id": 12345678, "file_path": "path/to/file.pdf"},
        {"document_id": 87654321, "file_path": "path/to/other.md"}
        ]


        - dict style:
        {
        "12345678": "path/to/file.pdf",
        "87654321": "path/to/other.md"
        }


        Raises:
            JsonStyleError: if bad json style.


        Returns:
            dict[int, Path]: the pre-validation object.


        """
        with self.mapping_file_path.open() as f:
            data = json.load(f)

        result = {}

        # list[dict] (array) style
        if isinstance(data, list):
            logger.debug("json format is array.")
            for item in data:
                doc_id = int(item["document_id"])
                file_path = self._resolve_file_path(item["file_path"])
                result[doc_id] = Path(file_path)

        # dict style
        elif isinstance(data, dict):
            logger.debug("json format is dict.")
            for doc_id_str, file_path in data.items():
                doc_id = int(doc_id_str)
                file_path_out = self._resolve_file_path(file_path)
                result[doc_id] = Path(file_path_out)

        else:
            bad_json = "json must be either a list(array) or dict format."
            raise JsonStyleError(bad_json)

        return result

    def _load_csv(self) -> dict[int, Path]:
        result = {}

        with self.mapping_file_path.open(newline="") as f:
            reader = csv.DictReader(f)

            # ensure required columns exist
            if (
                reader.fieldnames is None
                or "document_id" not in reader.fieldnames
                or "file_path" not in reader.fieldnames
            ):
                required_cols = "csv must contain 'document_id' and 'file_path' columns"
                raise ValueError(required_cols)

            for row in reader:
                doc_id = int(row["document_id"])
                file_path = self._resolve_file_path(row["file_path"])
                result[doc_id] = Path(file_path)

        return result

    def _resolve_file_path(self, file_path: str | Path) -> Path:
        """
        Resolve file path, handling absolute
        paths and document_base_dir.


        Args:
            file_path: File path from mapping file
            (can be relative or absolute)


        Returns:
            Resolved Path object


        Raises:
            FileNotFoundError: If resolved path doesn't exist


        """
        file_path = Path(file_path)

        # if already absolute, use as-is
        if file_path.is_absolute():
            resolved_path = file_path
            logger.debug(f"Using absolute path: {resolved_path}")

        # if document_base_dir provided, prepend it
        elif self.document_base_dir:
            resolved_path = self.document_base_dir / file_path
            logger.debug(f"Resolved relative path with base dir: {resolved_path}")

        # else relative path from cwd
        else:
            resolved_path = file_path.resolve()
            logger.debug(f"Resolved path from current directory: {resolved_path}")

        if not resolved_path.is_file():
            no_file = f"File not found: {resolved_path} (original: {file_path})"
            raise FileNotFoundError(no_file)

        return resolved_path


class DocumentReferenceLinker:
    """Core class for linking references/citations with parsed document text."""

    LINKING_STRATEGY_HIERARCHY = list(LinkingStrategy)  # retain the enum's order

    def __init__(
        self,
        references: list[Document],
        document_reference_mapping: list[DocumentReferenceMapping] | Path | None = None,
        document_base_dir: Path | None = None,
        parser: DocumentParser = parser,
        linking_strategies: list[LinkingStrategy] | None = None,
    ) -> None:
        """Initialise DocumentReferenceLinker class."""
        # deep copies to ensure lookup tables don't modify validators
        # on our existing reference docuemnts.
        tmp_refs = [doc.model_copy(deep=True) for doc in references]
        self.documents_references = references

        # lookup dics for O(1) id & author_year based lookup
        self._references_by_id: dict[int, Document] = {
            doc.document_id: doc for doc in tmp_refs if doc.document_id is not None
        }
        try:
            self._references_by_author_year_longest: dict[str, Document] = {
                doc.author_year_from_document_identity(
                    substring_strategy="longest"
                ): doc
                for doc in tmp_refs
            }
        except ValueError:
            logger.warning("author or year missing, returning empty dict.")
            self._references_by_author_year_longest = {}

        try:
            self._references_by_author_year_last: dict[str, Document] = {
                doc.author_year_from_document_identity(substring_strategy="last"): doc
                for doc in tmp_refs
            }
        except ValueError:
            logger.warning("author or year missing, returning empty dict.")
            self._references_by_author_year_last = {}

        self.document_base_dir = document_base_dir
        if isinstance(document_reference_mapping, Path):
            document_reference_mapping = MappingImporter(
                mapping_file_path=document_reference_mapping,
                document_base_dir=document_base_dir,
            ).import_mapping()
        self.document_reference_mapping = document_reference_mapping
        self.parser = parser

        self.linking_strategies = linking_strategies or self.LINKING_STRATEGY_HIERARCHY

    def _create_linking_factory(
        self, linking_strategy: LinkingStrategy
    ) -> Callable[[], Generator[LinkedInterimPayload, None, None]]:
        """
        Create a match function contingent on linking strategy.


        Args:
            linking_strategy (LinkingStrategy): see the enum for available ones.


        Returns:
            function: the function to run to retrieve objects to be linked,
            which will always yield a LinkedInterimPayload.


        """
        linking_strategy_map: dict[LinkingStrategy, Callable] = {  # extend as needed.
            LinkingStrategy.MAPPING_FILE: self._get_linkages_mapping_file,
            LinkingStrategy.FILENAME_AUTHOR_YEAR_LONGEST: partial(
                self._get_linkages_filename_author_year, "longest"
            ),
            LinkingStrategy.FILENAME_AUTHOR_YEAR_LAST: partial(
                self._get_linkages_filename_author_year, "last"
            ),
            LinkingStrategy.FILENAME_ID: self._get_linkages_filename_id,
        }

        return linking_strategy_map[linking_strategy]

    def _get_linkages_mapping_file(self) -> Generator[LinkedInterimPayload]:
        """
        Yield linkages between files (pdf/md) and document_ids.


        This is contingent on
        - successful reading of a mapping file (csv/json) and
          storing in self.document_reference_mapping
        - there being at least 1 document that matches the ids
          therein.


        Yields:
             Generator[LinkedInterimPayload].

        """
        if (
            self.document_reference_mapping is not None
            and False
            in [
                isinstance(x, DocumentReferenceMapping)
                for x in self.document_reference_mapping
            ]
        ) or self.document_reference_mapping is None:
            bad_mapping = (
                "self.document_reference_mapping needs to be a list of "
                " DocumentReferenceMapping objects."
                f"actual: {type(self.document_reference_mapping)}"
            )
            raise TypeError(bad_mapping)

        for mapping in self.document_reference_mapping:
            # get the right unlinked doc (reference)
            unlinked_doc = self._references_by_id.get(mapping.document_id)
            if unlinked_doc is None:
                logger.debug(
                    f"no reference document found for id {mapping.document_id}. next!"
                )
                continue
            interim_payload_dict = {
                **mapping.model_dump(),
                "unlinked_document": unlinked_doc,
            }
            yield LinkedInterimPayload(**interim_payload_dict)

    def _get_linkages_filename_author_year(
        self, substring_strategy: Literal["longest", "last"]
    ) -> Generator[LinkedInterimPayload]:
        """
        Yield linkages between files and reference-documents based on
        best guess at `author_year.pdf` filename structure.


        Yields:
            Generator[LinkedInterimPayload]:


        """
        # we could type-check or existence-check self.document_reference_mapping
        # here; but there might be an instance in which the user chooses this
        # regardless of other situation, so really this should be handled elsewhere.
        # necessary condition for this method to atempt is only availability of
        # files.
        if not self.document_base_dir or not self.document_base_dir.is_dir():
            bad_base_dir = "self.document_base_dir needs to be a valid directory."
            raise ValueError(bad_base_dir)

        lookup_dict: dict[str, Document]
        if substring_strategy == "longest":
            lookup_dict = self._references_by_author_year_longest
        elif substring_strategy == "last":
            lookup_dict = self._references_by_author_year_last
        else:
            bad_strat_err = "unimnplemented substring strategy for finding author"
            raise NotImplementedError(bad_strat_err)
        logger.debug(f"last name strategy is {substring_strategy}")

        for file in self.document_base_dir.iterdir():
            if file.suffix not in [".md", ".pdf"]:
                logger.warning(f"file {file} is not pdf/md. next!")
                continue
            author_year_guess = file.name.split(".")[0].lower()
            unlinked_doc = lookup_dict.get(author_year_guess)
            if unlinked_doc is None:
                logger.debug(
                    f"no reference document found for id {author_year_guess}. next!"
                )
                continue
            if unlinked_doc.document_identity is None:
                unlinked_doc.init_document_identity()

            yield LinkedInterimPayload(
                document_id=unlinked_doc.document_identity.document_id,  # type:ignore[union-attr, arg-type]
                file_path=file,
                format=cast("Literal['md', 'pdf']", file.suffix[1:]),
                unlinked_document=unlinked_doc,
            )

    def _get_linkages_filename_id(self) -> Generator[LinkedInterimPayload]:
        """
        Yield linkages between files and reference-documents based on
        assumption that files are named `id.pdf`, e.g `12345678.pdf`.


        Yields:
            Generator[LinkedInterimPayload]:


        """
        if not self.document_base_dir or not self.document_base_dir.is_dir():
            bad_base_dir = "self.document_base_dir needs to be a valid directory."
            raise ValueError(bad_base_dir)

        for file in self.document_base_dir.iterdir():
            if file.suffix not in [".md", ".pdf"]:
                logger.warning(f"file {file} is not pdf/md. next!")
                continue
            id_guess = int(file.name.split(".")[0])
            unlinked_doc = self._references_by_id.get(id_guess)
            if unlinked_doc is None:
                logger.debug(f"no reference document found for id {id_guess}. next!")
                continue
            if unlinked_doc.document_identity is None:
                unlinked_doc.init_document_identity()
            yield LinkedInterimPayload(
                document_id=unlinked_doc.document_identity.document_id,  # type:ignore[union-attr, arg-type]
                file_path=file,
                format=cast("Literal['md', 'pdf']", file.suffix[1:]),
                unlinked_document=unlinked_doc,
            )

    @staticmethod
    def _parse_pdf(
        path_to_pdf: Path, *, return_images: bool = False, return_metadata: bool = False
    ) -> ParsedOutput:
        """
        Parse a pdf to ParsedOutput.


        NOTE: wrapper around DocumentParser.parse().


        Args:
            path_to_pdf (Path): where to find the file
            return_images (bool, optional): store images or not. Defaults to False.
            return_metadata (bool, optional): store metadata or not. Defaults to False.


        Raises:
            TypeError: when it's not a pdf file.


        Returns:
            ParsedOutput: a container for markdown, images and metadata.


        """
        if path_to_pdf.suffix != ".pdf":
            not_pdf = "need a valid pdf file."
            raise TypeError(not_pdf)
        return parser(
            input_=path_to_pdf,
            return_images=return_images,
            return_metadata=return_metadata,
        )

    @staticmethod
    def link_reference_parsed_document(
        reference: Document,
        parsed_output: ParsedOutput,
        original_filepath: Path | None = None,
    ) -> Document:
        """
        Link a reference, in `Document` format with a parsed document,
        in ParsedOutput format.


        Args:
            reference (Document): the reference, e.g. 'document' from eppi json.
            parsed_output (ParsedOutput): parser output.
            original_filepath (Path | None, optional): Defaults to None.


        Returns:
            Document: a linked Document with
            all required fields populated, and is_linked==True,
            and required fields' presence validated.


        """
        if not reference.document_identity:
            logger.debug(
                "initialising document identity for "
                f"reference with id {reference.document_id}."
            )

            reference.init_document_identity()

        logger.debug(
            f"adding parsed_output to reference with id {reference.document_id}."
        )
        reference.link_parsed_document(
            parsed_document=parsed_output,
            original_doc_filepath=original_filepath,
        )

        # set context & context-type
        reference.set_context_from_parsed()
        # reference.context_type = ContextType.FULL_DOCUMENT

        # set is_linked -- this will raise a ValidationError
        # if min requirements are not met
        reference.is_final = True
        reference.is_linked = True

        return reference

    def link_many_references_parsed_documents(
        self,
        *,
        return_images: bool = False,
        return_metadata: bool = False,
    ) -> list[Document]:
        """
        Link multiple references to parsed
        documents using available LinkingStrategy(s).


        Iterates over linking strategies in
        hierarchical order, attempting to link
        each reference-doc to its corresponding file.


        If required, parses files.
        Creates Document objects where is_linked=True.


        Args:
            return_images: Whether to include images in parsed output
            return_metadata: Whether to include metadata in parsed output


        Returns:
            List of Document objects successfully linked and parsed


        """
        linked_documents: list[Document] = []
        processed_doc_ids = set()
        n_docs_to_link = len(self.documents_references)

        for strategy in self.linking_strategies:
            if len(linked_documents) == n_docs_to_link:
                logger.info("all linking jobs completed.")
                break
            logger.info(f"Attempting linking strategy: {strategy}")

            try:
                linking_function = self._create_linking_factory(strategy)  # type:ignore[arg-type] #mypy is wrong here...

                for interim_payload in linking_function():
                    if interim_payload.document_id in processed_doc_ids:
                        logger.debug(
                            f"document {interim_payload.document_id} "
                            " already linked, next"
                        )
                        continue

                    # parse only if pdf.
                    if interim_payload.format == "pdf":
                        parsed_output = self._parse_pdf(
                            interim_payload.file_path,
                            return_images=return_images,
                            return_metadata=return_metadata,
                        )
                    elif interim_payload.format == "md":
                        parsed_output = ParsedOutput(
                            text=interim_payload.file_path.read_text(encoding="utf-8"),
                            parser_library="unknown",
                        )
                    else:
                        logger.warning(
                            f"unsupported format {interim_payload.format} for "
                            f"document {interim_payload.document_id}, skipping"
                        )
                        continue

                    # link!
                    linked_doc = self.link_reference_parsed_document(
                        reference=interim_payload.unlinked_document,
                        parsed_output=parsed_output,
                        original_filepath=interim_payload.file_path,
                    )

                    linked_documents.append(linked_doc)
                    processed_doc_ids.add(interim_payload.document_id)

                    logger.info(
                        f"successfully linked document {interim_payload.document_id} "
                        f"with file {interim_payload.file_path} "
                        f"using {strategy}"
                    )

            except (TypeError, ValueError) as e:
                # if i + 1 < len(self.linking_strategies):
                logger.error(f"Error with linking strategy {strategy}: {e}")
                continue
                # raise

        total_refs = len(self.documents_references)
        linked_count = len(linked_documents)
        logger.info(
            f"linking complete: {linked_count}/{total_refs} "
            "references successfully linked"
        )

        if linked_count < total_refs:
            unlinked_ids = [
                doc.document_id
                for doc in self.documents_references
                if doc.document_id not in processed_doc_ids
            ]
            logger.warning(f"Unlinked document IDs: {unlinked_ids}")

        return linked_documents
