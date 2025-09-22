"""Utilities for parsing input files (e.g. pdf) for documents into output files (e.g. md)."""

from collections.abc import Callable
from enum import Enum, StrEnum, auto
from os import PathLike
from pathlib import Path
from typing import Annotated, Any

import pypandoc
from loguru import logger
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
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
    PNG = auto()
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
    output_file_types = [OutputFileType.MD, OutputFileType.PNG, OutputFileType.JSON]

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
        logger.debug("about to run text_from_rendered")
        text, metadata, images = text_from_rendered(rendered)
        logger.debug("finished")
        # logger.debug(f"text: {text}")
        logger.debug(f"metadata: {metadata}")
        logger.debug(f"image: {images}")
        logger.debug(f"return metadata : {return_metadata}")
        logger.debug(f"return images: {return_images}")
        out = [text]
        if return_metadata:
            out.append(metadata)
        if return_images:
            out.append(images)
        # logger.debug(f"out: {out}")
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
            return_images (bool): Defaults to False. Whether to write parsed images (png) to file, or not. `out_path`
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
            parser = self.__getattribute__(f"default_parser_{input_file_type.value}")
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
        logger.debug(f"managed to exit self.parse.")

        if check_language and (
            (isinstance(parsed, tuple) and not self.check_text_is_english(parsed[0]))
            or (isinstance(parsed, str) and not self.check_text_is_english(parsed))
        ):
            bad_english_error = f"{input_file} was not parsed with good English."
            raise BadEnglishError(bad_english_error)

        # if out_path:
        #     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        #     if Path(out_path).is_file():

        #     with Path(output_file).open("w") as outfile:
        #         outfile.write(parsed_text)
        #         return str(output_file)

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
        if return_images and OutputFileType.PNG not in parser.output_file_types:
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

    # @staticmethod
    # def parse_pdf(
    #     file: str | PathLike,
    #     parser: ParserLibrary,
    #     *,
    #     return_images: bool = False,
    #     return_metadata: bool = False,
    # ) -> str | tuple[str, Any, Any]:
    #     """
    #     Parse pdf file to string in md format.

    #     Args:
    #         file (str | PathLike): Path to pdf file.
    #         parser (ParserLibrary): The Parser to use.

    #     Raises:
    #         FileParserMismatchError: If an illegal parser was selected.

    #     Returns:
    #         str: The text, as str, formatted as markdown.

    #     """
    #     logger.debug(f"parsing file {file} using parser {parser}...")
    #     if parser.name == "marker":
    #         rendered = converter(file)
    #         text, metadata, images = text_from_rendered(rendered)
    #         logger.debug("successfully parsed pdf.")
    #         out = text

    #     if return_metadata:
    #         out = out, metadata
    #     if return_images:
    #         out = out, images

    #     return out

    # @staticmethod
    # def parse_epub(file: str | PathLike, parser: ParserLibrary) -> str:
    #     """
    #     Parse epub file to string in md format.

    #     Args:
    #         file (str | PathLike): Path to epub file.
    #         parser (ParserLibrary): The Parser to use.

    #     Raises:
    #         FileParserMismatchError: If an illegal parser was selected.

    #     Returns:
    #         str: The text, as str, formatted as markdown.

    #     """
    #     logger.debug(f"parsing file {file} using parser {parser}...")
    #     if parser.name == "pandoc":
    #         logger.debug("successfully parsed epub.")
    #         return pypandoc.convert_file(file, to="md", format="epub")
    #     bad_parser_file_combo = f"Unsupported parser {parser} for file {file}."
    #     raise FileParserMismatchError(bad_parser_file_combo)

    # @staticmethod
    # def parse_html(file: str | PathLike, parser: ParserLibrary) -> str:
    #     """
    #     Parse html file to string in md format.

    #     Args:
    #         file (str | PathLike): Path to html file.
    #         parser (ParserLibrary): The Parser to use.

    #     Raises:
    #         FileParserMismatchError: If an illegal parser was selected.

    #     Returns:
    #         str: The text, as str, formatted as markdown.

    #     """
    #     logger.debug(f"parsing file {file} using parser {parser}...")
    #     if parser.name == "pandoc":
    #         logger.debug("successfully parsed html.")
    #         return pypandoc.convert_file(file, to="md", format="html")
    #     bad_parser_file_combo = f"Unsupported parser {parser} for file {file}."
    #     raise FileParserMismatchError(bad_parser_file_combo)

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
