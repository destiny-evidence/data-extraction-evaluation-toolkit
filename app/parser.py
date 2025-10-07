"""Utilities for parsing input files (e.g. pdf) for documents into output files (e.g. md)."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum, auto
from os import PathLike
from pathlib import Path

import pypandoc
from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from PIL.Image import Image
from pydantic import BaseModel, field_validator

from app.assess_text_quality import check_language
from app.exceptions import (
    FileParserMismatchError,
    InvalidFileTypeError,
    InvalidInputFileTypeError,
    InvalidOutputFileTypeError,
)

# init marker converter
artifact_dict = create_model_dict()
converter = PdfConverter(artifact_dict=artifact_dict)


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
    """

    text: str
    images: dict[str, Image] | None = None
    metadata: dict | None = None

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

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
    input_file_types: list[InputFileType]
    output_file_types: list[OutputFileType]

    @classmethod
    @abstractmethod
    def parse(
        cls,
        input_file: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        """
        Parse a document.
        Intentionelly left blank as this should be populated in sub-classes.

        Args:
            input_file (str | PathLike): Path to input file.
            return_metadata (bool, optional): Return json metadata. Defaults to False.
            return_images (bool, optional): Return images in document. Defaults to False.

        Raises:
            NotImplementedError: The default, should never actually come.

        Returns:
            str | tuple[str, Any, Any]: There will always be str, but sometimes more.

        """
        raise NotImplementedError


class MarkerParser(ParserLibrary):
    """Parser with `marker` backend."""

    name = "marker"
    input_file_types = [InputFileType.PDF]
    output_file_types = [OutputFileType.MD, OutputFileType.JPEG, OutputFileType.JSON]

    @classmethod
    def parse(
        cls,
        input_file: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,  # noqa: ARG003
    ) -> ParsedOutput:
        """Parse file using marker."""
        rendered = converter(input_file)
        text, extension, images = text_from_rendered(rendered)
        out = {"text": text}
        if return_metadata:
            out["metadata"] = rendered.metadata
        if return_images:
            out["images"] = images
        return ParsedOutput(**out)


class PandocParser(ParserLibrary):
    """Parser with `pandoc` backend."""

    name = "pandoc"
    input_file_types = [InputFileType.EPUB, InputFileType.HTML, InputFileType.XML]
    output_file_types = [OutputFileType.MD]

    @classmethod
    def parse(
        cls,
        input_file: str | PathLike,
        input_file_type: InputFileType | str | None = None,
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
        if input_is_string and not input_file_type:
            missing_filetype = (
                "if input is str in memory, provide format as `input_file_type`."
            )
            raise InvalidInputFileTypeError(missing_filetype)
        if not input_file_type:
            input_file_type = DocumentParser.detect_filetype(
                input_file, cls.input_file_types
            )
        if isinstance(input_file_type, InputFileType):
            input_file_type = input_file_type.value
        if input_file_type == "xml":
            input_file_type = "jats"

        if input_is_string:
            parse_method = pypandoc.convert_text
        else:
            parse_method = pypandoc.convert_file

        out = {
            "text": parse_method(
                input_file,
                to="md",
                format=input_file_type,
            )
        }
        return ParsedOutput(**out)


class DocumentParser:
    """Parse documents from target format to other target format."""

    DEFAULT_PARSERS: dict[str, type[ParserLibrary]] = {
        "pdf": MarkerParser,
        "epub": PandocParser,
        "html": PandocParser,
        "xml": PandocParser,
    }

    def __init__(
        self, parsers: dict[str, type[ParserLibrary]] = DEFAULT_PARSERS
    ) -> None:
        """
        Initialise instance of DocumentParser with default parsers.

        Args:
            default_parser_pdf (ParserLibrary, optional): _description_. Defaults to MARKER_PARSER.
            default_parser_epub (ParserLibrary, optional): _description_. Defaults to PANDOC_PARSER.
            default_parser_html (ParserLibrary, optional): _description_. Defaults to PANDOC_PARSER.

        """
        self.parsers = parsers

        if self.parsers is not None and isinstance(self.parsers, dict):
            for parser_name, parser in self.parsers.items():
                logger.debug(f"default {parser_name} parser: {parser.name}")

    def __call__(  # noqa: PLR0913 - img/meta needs to be explicit (non-kwargs) here.
        self,
        input_file: str | PathLike,
        out_path: str | PathLike | None = None,
        parser: type[ParserLibrary] | None = None,
        input_file_type: InputFileType | str | None = None,
        *,
        return_images: bool = False,
        return_metadata: bool = False,
        **kwargs,
    ) -> ParsedOutput:
        """
        Run the parser on one input_file.

        Args:
            input_file (str | PathLike): _description_
            output_file (str | PathLike | None): If None, return parsed content as str.
            parser (ParserLibrary | None, optional): _description_. Defaults to None. If None,
                                                     uses the default parser.
            input_file_type (InputFileType | None, optional): _description_. Defaults to None.
                                                             If None, infers file type using `detect_filetype`.
            return_images (bool): Defaults to False. Whether to write parsed images (JPEG) to file, or not. `out_path`
                                can't be None.
            return_metadata (bool): Defaults to None. Whether to write parsed metadata (json).

        Returns:
            str: _description_

        """
        logger.debug(f"kwargs: {kwargs}")
        if input_file_type is None:
            logger.debug(
                "no input file type provided. using `detect_filetype` to infer."
            )
            try:
                input_file_type = InputFileType(
                    self.detect_filetype(
                        file=input_file,
                        permitted_file_enum_list=list(InputFileType),
                    )
                )
            except ValueError as ve:
                raise InvalidInputFileTypeError(ve) from ve
        logger.debug(f"input file type: {input_file_type}.")

        if parser is not None and (
            (not isinstance(parser, type)) or (not issubclass(parser, ParserLibrary))
        ):
            bad_parser_err = f"parser {parser} is not a valid ParserLibrary."
            raise FileParserMismatchError(bad_parser_err)
        if parser is None and input_file_type is not None:
            logger.debug("parser not supplied. selecting default parser for file_type.")
            if isinstance(input_file_type, str):
                input_file_type = InputFileType(input_file_type)
            if self.parsers is None or input_file_type.value not in self.parsers:
                missing_parser = "no parser supplied."
                raise ValueError(missing_parser)
            parser = self.parsers[input_file_type.value]
        if parser is None or (
            parser is None and input_file_type is None
        ):  # for pedantic mypy
            missing_parser = "no parser supplied."
            raise ValueError(missing_parser)
        logger.debug(f"parser: {parser}.")
        kwargs["input_file_type"] = input_file_type

        parsed = self.parse(
            input_file=input_file,
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
        input_file: str | PathLike,
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
            input_file (str | PathLike): _description_
            input_file_type (InputFileType): _description_
            parser (ParserLibrary): _description_
            parse_method (Callable[[str  |  PathLike, ParserLibrary], str]): _description_

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
            input_file=input_file,
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

            forbidden_file_type = f"file type {extension} is not permitted. Use one of {permitted_extensions_str}."
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
