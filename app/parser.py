"""Utilities for parsing input files (e.g. pdf) for documents into output files (e.g. md)."""

import json
from collections.abc import Callable
from enum import Enum, StrEnum, auto
from os import PathLike
from pathlib import Path
from typing import Annotated, Any

import pypandoc
from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import convert_if_not_rgb, text_from_rendered
from PIL.Image import Image
from pydantic import BaseModel, Field

from app.assess_text_quality import is_english
from app.exceptions import (
    BadEnglishError,
    FileParserMismatchError,
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


class OutputFileType(StrEnum):
    """
    Enumeration of permitted output file types.

    Args:
        StrEnum (_type_):

    """

    MD = auto()
    JPEG = auto()
    JSON = auto()


class ParserLibrary:
    """Base parser class."""

    name: str
    input_file_types: list[InputFileType]
    output_file_types: list[OutputFileType]

    @classmethod
    def parse(
        cls,
        input_file: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> str | tuple[str, Any, Any]:
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
        **kwargs,
    ) -> str | tuple[str, Any, Any]:
        """Parse file using marker."""
        rendered = converter(input_file)
        text, extension, images = text_from_rendered(rendered)
        out = [text]
        if return_metadata:
            out.append(rendered.metadata)
        if return_images:
            out.append(images)
        return tuple(out)


class PandocParser(ParserLibrary):
    """Parser with `pandoc` backend."""

    name = "pandoc"
    input_file_types = [InputFileType.EPUB, InputFileType.HTML]
    output_file_types = [OutputFileType.MD]

    @classmethod
    def parse(
        cls,
        input_file: str | PathLike,
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> str:
        """Parse file using pandoc."""
        if True in [return_images, return_metadata]:
            # NOTE: this is probably not the best method, as it doesn't
            # reference input or output file types. can we think of a generic
            # method for doing this via ParserLibrary, without using pydantic?
            image_meta_erro = "PandocParser can't produce images or metadata."
            raise InvalidOutputFileTypeError(image_meta_erro)
        # detect input file type as explicitly supplying it would be
        # cumbersome
        input_file_type = DocumentParser.detect_filetype(input_file)
        # currently only markdown output
        return pypandoc.convert_file(
            input_file,
            to="md",
            format=input_file_type.value,
        )


class DocumentParser:
    """Parse documents from target format to other target format."""

    def __init__(
        self,
        default_parser_pdf: type[ParserLibrary] = MarkerParser,
        default_parser_epub: type[ParserLibrary] = PandocParser,
        default_parser_html: type[ParserLibrary] = PandocParser,
    ) -> None:
        """
        Initialise instance of DocumentParser with default parsers.

        Args:
            default_parser_pdf (ParserLibrary, optional): _description_. Defaults to MARKER_PARSER.
            default_parser_epub (ParserLibrary, optional): _description_. Defaults to PANDOC_PARSER.
            default_parser_html (ParserLibrary, optional): _description_. Defaults to PANDOC_PARSER.

        """
        self.default_parser_epub = default_parser_epub
        self.default_parser_html = default_parser_html
        self.default_parser_pdf = default_parser_pdf

        logger.debug(f"default epub parser: {self.default_parser_epub}")
        logger.debug(f"default html parser: {self.default_parser_html}")
        logger.debug(f"default pdf parser: {self.default_parser_pdf}")

    def __call__(  # noqa: PLR0913 - img/meta needs to be explicit (non-kwargs) here.
        self,
        input_file: str | PathLike,
        out_path: str | PathLike | None = None,
        parser: type[ParserLibrary] | None = None,
        input_file_type: InputFileType | None = None,
        *,
        return_images: bool = False,
        return_metadata: bool = False,
        check_language: bool = False,
        **kwargs,
    ) -> str | tuple[str, Any, Any]:
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
        if not input_file_type:
            logger.debug(
                "no input file type provided. using `detect_filetype` to infer."
            )
            input_file_type = self.detect_filetype(input_file)
        if input_file_type not in InputFileType:
            input_file_type_not_permitted = (
                f"input_file_type {input_file_type} is not permitted."
            )
            raise InvalidInputFileTypeError(input_file_type_not_permitted)
        logger.debug(f"input file type: {input_file_type}.")

        if not parser:
            logger.debug("parser not supplied. selecting default parser for file_type.")
            parser: ParserLibrary = self.__getattribute__(  # type: ignore[no-redef]
                f"default_parser_{input_file_type.value}"
            )
        if parser is None:  # for pedantic mypy
            missing_parser = "no parser supplied."
            raise ValueError(missing_parser)
        logger.debug(f"parser: {parser}.")

        parsed = self.parse(
            input_file=input_file,
            parser=parser,
            return_images=return_images,
            return_metadata=return_metadata,
            **kwargs,
        )

        if check_language and (
            (isinstance(parsed, tuple) and not self.check_text_is_english(parsed[0]))
            or (isinstance(parsed, str) and not self.check_text_is_english(parsed))
        ):
            bad_english_error = f"{input_file} was not parsed with good English."
            raise BadEnglishError(bad_english_error)

        if out_path:
            images = None
            metadata = None
            text = parsed[0]
            if return_images and return_metadata:
                metadata = parsed[1]
                images = parsed[2]
            if return_images and not return_metadata:
                images = parsed[1]
            if not return_images and return_metadata:
                metadata = parsed[1]

            self.write_files(
                out_path=out_path,
                parser=parser,
                write_metadata=return_metadata,
                write_images=return_images,
                text=text,
                metadata=metadata,  # type: ignore[arg-type]
                images=images,  # type: ignore[arg-type]
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
    ) -> str | tuple[str, Any, Any]:
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
    def detect_filetype(file: str | PathLike) -> InputFileType:
        """
        Detect file type from a file_path.

        Args:
            file (str | PathLike): _description_

        Raises:
            InvalidInputFileTypeError: If file extension isn't permitted.

        Returns:
            InputFileType: _description_

        """
        extension = str(file).split(".")[-1]
        if extension not in InputFileType:
            forbidden_file_type = f"file type {extension} is not permitted. Use one of {list(InputFileType)}."
            raise InvalidInputFileTypeError(forbidden_file_type)
        logger.debug(f"filetype is: {extension}.")
        return InputFileType(extension)

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

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(out_path).is_file() or (
            Path(out_path).name.split(".")[-1] in required_outfiles
        ):
            logger.debug(f"`out_path` {out_path} points to a file.")

            dir_base = Path(out_path).parent
            filename_base = "".join(Path(out_path).name.split(".")[0:-1])
            extension = Path(out_path).name.split(".")[-1]
            if (extension not in parser.output_file_types) or (
                extension not in required_outfiles
            ):
                bad_out_file_ext = (
                    f"supplied out filetype {extension}",
                    " is not permitted or not applicable ",
                    "for parser {parser.name}",
                )
                raise InvalidOutputFileTypeError(bad_out_file_ext)

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

    @staticmethod
    def check_text_is_english(parsed_text: str) -> bool:
        """
        Check if parsed text is intelligible English.

        Args:
            parsed_text (str): the parsed text as string.

        Returns:
            bool: Indicating whether it's 'proper' English or not.

        """
        return is_english(parsed_text)
