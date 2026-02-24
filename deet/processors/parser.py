"""Utilities for parsing input files (e.g. pdf) into output files (e.g. md)."""

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import StrEnum, auto
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Literal

import pypandoc
from diskcache import Cache
from loguru import logger
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from deet.exceptions import (
    EmptyPdfExtractionError,
    FileParserMismatchError,
    InvalidFileTypeError,
    InvalidInputFileTypeError,
    InvalidOutputFileTypeError,
)
from deet.settings import get_settings
from deet.utils.assess_text_quality import check_language

# CACHE init
CACHE_DIR = get_settings().base_disk_cache_dir / "marker_parser_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
parser_cache = Cache(str(CACHE_DIR))


class InputFileType(StrEnum):
    """
    Enumeration of permitted input file types.

    Args:
        StrEnum (_type_):

    """

    PDF = auto()
    EPUB = auto()
    HTML = auto()
    XML = auto()  # NOTE - this only covers JATS xml.


class OutputFileType(StrEnum):
    """
    Enumeration of permitted output file types.

    Args:
        StrEnum (_type_):

    """

    MD = auto()
    JPEG = auto()
    JSON = auto()


class ParsedOutput(BaseModel):
    """
    Output returned from the `parser()` method of subclasses of ParserLibrary.

    Contains:
        text, str: md-formatted parsed text (required)
        images, pillow.img: pillow-formatted image(s) (optional)
        metadata, dict: metadata json (optional)
        timestamp: datetime: auto-populates with _now_
        parser_library: str: name of the ParserLibrary implementation used
    """

    text: str
    images: dict[str, Image] | None = None
    metadata: dict | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    parser_library: Literal[
        "pandoc", "marker", "pdfminer"
    ]  # extend when adding new parsers

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # this is to allow our Executor class as a type.

    @field_validator("text", mode="after")
    @classmethod
    def assess_language_quality(cls, value: str) -> str:
        """
        Assess language quality.

        Args:
            text (str): Parsed text.

        Raises:
            MalformedLanguageError: If threshold not met.

        Returns:
            str: parsed text.

        """
        if not check_language(value):
            logger.debug("check lang failed")
            bad_language = "Supplied text didn't pass quality check."
            raise ValueError(bad_language)
        return value


class ParserLibrary(ABC):
    """Base parser class."""

    name: str
    input_types: list[InputFileType]
    output_file_types: list[OutputFileType]

    @classmethod
    @abstractmethod
    def parse(
        cls,
        input_: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        """
        Parse a document.
        Intentionelly left blank as this should be populated in sub-classes.

        Args:
            input_ (str | PathLike): Path to input file or string of input string.
            return_metadata (bool, optional): Return json metadata. Defaults to False.
            return_images (bool, optional): Return images in doc. Defaults to False.

        Raises:
            NotImplementedError: The default, should never actually come.

        Returns:
            str | tuple[str, Any, Any]: There will always be str, but sometimes more.

        """
        raise NotImplementedError


class MarkerParser(ParserLibrary):
    """Parser with `marker` backend."""

    name: Literal["marker"] = "marker"
    input_types = [InputFileType.PDF]
    output_file_types = [OutputFileType.MD, OutputFileType.JPEG, OutputFileType.JSON]

    @classmethod
    @parser_cache.memoize(typed=True, expire=None, tag="marker-converter")
    def _get_converter(cls):  # noqa: ANN206 no return type hint as we don't know PdfConverter yet
        """Lazy initialization of marker converter with disk caching."""
        logger.debug("Initializing marker converter...")
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        artifact_dict = create_model_dict()
        return PdfConverter(artifact_dict=artifact_dict)

    @classmethod
    def parse(
        cls,
        input_: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,  # noqa: ARG003
    ) -> ParsedOutput:
        """Parse file using marker."""
        from marker.output import text_from_rendered

        converter = cls._get_converter()
        rendered = converter(str(input_))
        text, extension, images = text_from_rendered(rendered)
        out = {"text": text}
        if return_metadata:
            out["metadata"] = rendered.metadata
        if return_images:
            out["images"] = images
        return ParsedOutput(**out, parser_library=cls.name)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the cached converter."""
        parser_cache.delete("marker-converter")
        logger.info("Marker converter cache cleared")


class PdfminerParser(ParserLibrary):
    """Parser with pdfminer.six backend. Fast text extraction, no images or metadata."""

    name: Literal["pdfminer"] = "pdfminer"
    input_types = [InputFileType.PDF]
    output_file_types = [OutputFileType.MD]
    _LAPARAMS = LAParams()

    @classmethod
    def parse(
        cls,
        input_: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,  # noqa: ARG003
    ) -> ParsedOutput:
        """Parse file using pdfminer.six (no OCR)."""
        if return_metadata or return_images:
            msg = "PdfminerParser can't produce images or metadata."
            raise InvalidOutputFileTypeError(msg)
        rsrcmgr = PDFResourceManager()
        out = StringIO()
        device = TextConverter(rsrcmgr, out, laparams=cls._LAPARAMS)
        with Path(input_).open("rb") as f:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.get_pages(f):
                interpreter.process_page(page)
        device.close()
        text = out.getvalue() or ""
        if not text.strip():
            raise EmptyPdfExtractionError(EmptyPdfExtractionError.DEFAULT_MESSAGE)
        return ParsedOutput(text=text, parser_library=cls.name)


class PandocParser(ParserLibrary):
    """Parser with `pandoc` backend."""

    name: Literal["pandoc"] = "pandoc"
    input_types = [InputFileType.EPUB, InputFileType.HTML, InputFileType.XML]
    output_file_types = [OutputFileType.MD]

    @classmethod
    def parse(
        cls,
        input_: str | PathLike,
        input_type: InputFileType | str | None = None,
        *,
        input_is_string: bool = False,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,  # noqa: ARG003
    ) -> ParsedOutput:
        """Parse file using pandoc."""
        if True in [return_images, return_metadata]:
            image_meta_erro = "PandocParser can't produce images or metadata."
            raise InvalidOutputFileTypeError(image_meta_erro)
        if input_is_string and not input_type:
            missing_filetype = (
                "if input is str in memory, provide format as `input_type`."
            )
            raise InvalidInputFileTypeError(missing_filetype)
        if not input_type:
            input_type = DocumentParser.detect_filetype(input_, cls.input_types)
        if isinstance(input_type, InputFileType):
            input_type = input_type.value
        if input_type == "xml":
            input_type = "jats"

        if input_is_string:
            parse_method = pypandoc.convert_text
        else:
            parse_method = pypandoc.convert_file

        out = {
            "text": parse_method(
                input_,
                to="md",
                format=input_type,
            )
        }
        return ParsedOutput(**out, parser_library=cls.name)


class DocumentParser:
    """Parse documents from target format to other target format."""

    DEFAULT_PARSERS: dict[str, type[ParserLibrary]] = {
        "pdf": PdfminerParser,
        "epub": PandocParser,
        "html": PandocParser,
        "xml": PandocParser,
    }

    def __init__(
        self, parsers: dict[str, type[ParserLibrary]] = DEFAULT_PARSERS
    ) -> None:
        """
        Initialise instance of DocumentParser with default parsers.
        Default parsers are in dict.

        """
        self.parsers = parsers

        if self.parsers is not None and isinstance(self.parsers, dict):
            for parser_name, parser in self.parsers.items():
                logger.debug(f"default {parser_name} parser: {parser.name}")

    def __call__(  # noqa: PLR0913 - img/meta needs to be explicit (non-kwargs) here.
        self,
        input_: str | PathLike,
        out_path: str | PathLike | None = None,
        parser: type[ParserLibrary] | None = None,
        input_type: InputFileType | str | None = None,
        *,
        return_images: bool = False,
        return_metadata: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        """
        Run the parser on one input_.

        Args:
            input_ (str | PathLike): File(path) or str of input_.
            out_path (str | PathLike | None): If None, return parsed content as str.
            parser (ParserLibrary | None, optional): Defaults to None.
                If None, uses the default parser.
            input_type (InputFileType | None, optional): Defaults to None.
                If None, infers file type using `detect_filetype`.
            return_images (bool): Defaults to False. Whether to write
                parsed images (JPEG) to file, or not. `out_path`. can't be None.
            return_metadata (bool): Defaults to None. Whether to write
                parsed metadata (json).

        Returns:
            str: ParsedOutput object.

        """
        logger.debug(f"kwargs: {kwargs}")
        if input_type is None:
            logger.debug(
                "no input file type provided. using `detect_filetype` to infer."
            )
            try:
                input_type = InputFileType(
                    self.detect_filetype(
                        file=input_,
                        permitted_file_enum_list=list(InputFileType),
                    )
                )
            except ValueError as ve:
                raise InvalidInputFileTypeError(ve) from ve
        logger.debug(f"input file type: {input_type}.")

        if parser is not None and (
            (not isinstance(parser, type)) or (not issubclass(parser, ParserLibrary))
        ):
            bad_parser_err = f"parser {parser} is not a valid ParserLibrary."
            raise FileParserMismatchError(bad_parser_err)
        if parser is None and input_type is not None:
            logger.debug("parser not supplied. selecting default parser for file_type.")
            if isinstance(input_type, str):
                try:
                    input_type = InputFileType(input_type)
                except ValueError as ve:
                    if "is not a valid InputFileType" in str(ve):
                        invalid_input_ft = f"{input_type} is not a valid InputFileType"
                        raise InvalidInputFileTypeError(invalid_input_ft) from ve
            if (
                self.parsers is None
                or (isinstance(input_type, str) and input_type not in self.parsers)
                or (
                    isinstance(input_type, InputFileType)
                    and input_type.value not in self.parsers
                )
            ):
                missing_parser = "no parser supplied."
                raise ValueError(missing_parser)
            if isinstance(input_type, InputFileType):
                parser = self.parsers[input_type.value]
            elif isinstance(input_type, str):
                parser = self.parsers[input_type]
            else:
                missing_parser = "no parser supplied."
                raise ValueError(missing_parser)
        logger.debug(f"parser: {parser}.")
        kwargs["input_type"] = input_type

        parsed = self.parse(
            input_=input_,
            parser=parser,
            return_images=return_images,
            return_metadata=return_metadata,
            **kwargs,
        )

        if out_path:
            self.write_files(
                out_path=out_path,
                parser=parser,
                write_metadata=return_metadata,
                write_images=return_images,
                text=parsed.text,
                metadata=parsed.metadata,
                images=parsed.images,
            )

        return parsed

    def parse(
        self,
        input_: str | PathLike,
        parser: type[ParserLibrary],
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        """
        Parse target file.
        Wraps around specific parser methods.

        Args:
            input_ (str | PathLike):
            input_type (InputFileType):
            parser (ParserLibrary):
            parse_method (Callable[[str  |  PathLike, ParserLibrary], str]):

        Returns:
            str: _description_

        """
        logger.debug(f"kwargs: {kwargs}")
        if return_metadata and OutputFileType.JSON not in parser.output_file_types:
            metadata_not_allowed = (
                f"metadata out not permitted for parser {parser.name}."
            )
            raise InvalidOutputFileTypeError(metadata_not_allowed)
        if return_images and OutputFileType.JPEG not in parser.output_file_types:
            images_not_allowed = f"images out not permitted for parser {parser.name}."
            raise InvalidOutputFileTypeError(images_not_allowed)

        return parser.parse(
            input_=input_,
            return_metadata=return_metadata,
            return_images=return_images,
            **kwargs,
        )

    @staticmethod
    def detect_filetype(
        file: str | PathLike,
        permitted_file_enum_list: list[InputFileType]
        | list[OutputFileType]
        | list[str]
        | None = None,
    ) -> str:
        """
        Detect file type from a file_path.

        Args:
            file (str | PathLike): _description_

        Raises:
            InvalidInputFileTypeError: If file extension isn't permitted.

        Returns:
            InputFileType: _description_

        """
        if permitted_file_enum_list is None:
            permitted_file_enum_list = list(InputFileType) + list(OutputFileType)
        permitted_extensions_str = {
            x.value
            for x in permitted_file_enum_list
            if isinstance(x, (InputFileType | OutputFileType))
        }

        extension = str(file).split(".")[-1]
        if extension not in permitted_extensions_str:
            has_input = any(
                isinstance(ft, InputFileType) for ft in permitted_file_enum_list
            )
            has_output = any(
                isinstance(ft, OutputFileType) for ft in permitted_file_enum_list
            )
            target_error: type[Exception]
            if has_input and not has_output:
                target_error = InvalidInputFileTypeError
            elif not has_input and has_output:
                target_error = InvalidOutputFileTypeError
            else:
                target_error = InvalidFileTypeError

            forbidden_file_type = (
                f"file type {extension} is not permitted. "
                f" Use one of {permitted_extensions_str}."
            )
            raise target_error(forbidden_file_type)

        logger.debug(f"filetype is: {extension}.")
        return extension

    @staticmethod
    def write_files(  # noqa: PLR0913
        out_path: str | PathLike,
        parser: type[ParserLibrary],
        *,
        write_metadata: bool,
        write_images: bool,
        text: str,
        metadata: dict | None = None,
        images: dict[str, Image] | None = None,
    ) -> None:
        """
        Write parsed content to file(s).

        NOTE: we are taking existence of `out_path` as an intention to
        write all requested objects to file. out_path can be a file or a dir.
        if out_path is a file, we write remaining files to parent dir.

        Args:
            out_path (str | PathLike): _description_
            write_metadata (bool): _description_
            write_images (bool): _description_
            text (str): _description_
            metadata (dict | None, optional): _description_. Defaults to None.
            images (dict[str, Image] | None, optional): _description_. Defaults to None.

        """
        extension = (
            DocumentParser.detect_filetype(  # should raise error if not permitted
                out_path, permitted_file_enum_list=parser.output_file_types
            )
        )

        required_outfiles = ["md"]
        if write_images:
            required_outfiles.append("jpeg")
            if images is None:  # or raise something?
                logger.warning(
                    "`write_images` set to True, but no images obj supplied."
                )
        if write_metadata:
            required_outfiles.append("json")
            if metadata is None:  # or raise something?
                logger.warning(
                    "`write_metadata` set to True, but no metadata obj supplied."
                )
        logger.debug(f"required outfiles: {required_outfiles}")
        if False in [ft in parser.output_file_types for ft in required_outfiles]:
            raise InvalidOutputFileTypeError

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(out_path).is_file() or (extension in required_outfiles):
            logger.debug(f"`out_path` {out_path} points to a file.")

            dir_base = Path(out_path).parent
            filename_base = "".join(Path(out_path).name.split(".")[0:-1])

        if Path(out_path).is_dir():
            logger.debug(f"`out_path` {out_path} points to a dir.")
            # we now have to get our filename base from somewhere...
            dir_base = Path(out_path)
            filename_base = text.split("\n")[0][:15].replace(" ", "_").lower()

        for ext in required_outfiles:
            out = dir_base / (filename_base + "." + ext)
            logger.debug(f"writing out {ext} to {out}.")
            if ext == "md":
                out.write_text(text)
            if ext == "json" and metadata is not None:
                out.write_text(json.dumps(metadata))
            if ext == "jpeg" and images is not None:
                for img_name, img in images.items():
                    img_out = dir_base / (filename_base + "_" + img_name)
                    img.save(img_out, ext)
