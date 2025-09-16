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


class ParserBase(BaseModel):
    """Generic data model for a Parser."""

    name: str
    input_file_types: list[InputFileType]
    output_file_types: list[OutputFileType]

    class Config:  # noqa: D106
        use_enum_values = True


class MarkerParser(ParserBase):
    """Data model for the marker parser."""

    name = "marker"
    input_file_types = ["pdf"]
    output_file_types = ["md", "png", "json"]


class PandocParser(ParserBase):
    """Data model for the pandoc parser."""

    name = "pandoc"
    input_file_types = ["epub", "html"]
    output_file_types = ["md"]


ParserLibrary = Annotated[
    MarkerParser | PandocParser, Field(discriminator="parser type")
]


class DocumentParser:
    """Parse documents from target format to other target format."""

    def __init__(
        self,
        default_parser_pdf: ParserLibrary = MarkerParser,
        default_parser_epub: ParserLibrary = PandocParser,
        default_parser_html: ParserLibrary = PandocParser,
    ) -> None:
        """
        Initialise instance of DocumentParser with default parsers.

        Args:
            default_parser_pdf (ParserLibrary, optional): _description_. Defaults to 'marker'.
            default_parser_epub (ParserLibrary, optional): _description_. Defaults to 'pandoc'.
            default_parser_html (ParserLibrary, optional): _description_. Defaults to 'pandoc'.

        """
        self.default_parser_epub = default_parser_epub
        self.default_parser_html = default_parser_html
        self.default_parser_pdf = default_parser_pdf

        logger.debug(f"default epub parser: {self.default_parser_epub}")
        logger.debug(f"default html parser: {self.default_parser_html}")
        logger.debug(f"default pdf parser: {self.default_parser_pdf}")

        self.input_file_parser_method_map = {
            InputFileType.PDF: self.parse_pdf,
            InputFileType.EPUB: self.parse_epub,
            InputFileType.HTML: self.parse_html,
        }

    def __call__(
        self,
        input_file: str | PathLike,
        out_path: str | PathLike | None = None,
        parser: ParserLibrary | None = None,
        input_file_type: InputFileType | None = None,
        *,
        save_images: bool = False,
        save_metadata: bool = False,
        check_language: bool = False,
        **kwargs,
    ) -> str:
        """
        Run the parser on one input_file.

        Args:
            input_file (str | PathLike): _description_
            output_file (str | PathLike | None): If None, return parsed content as str.
            parser (ParserLibrary | None, optional): _description_. Defaults to None. If None,
                                                     uses the default parser.
            input_file_type (InputFileType | None, optional): _description_. Defaults to None.
                                                             If None, infers file type using `detect_filetype`.
            save_images (bool): Defaults to False. Whether to write parsed images (png) to file, or not. `out_path`
                                can't be None.
            save_metadata (bool): Defaults to None. Whether to write parsed metadata (json).

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

        parse_method = self.input_file_parser_method_map.get(input_file_type)
        if parse_method is None:
            missing_parse_method = "no parse method supplied."
            raise InvalidInputFileTypeError(missing_parse_method)
        logger.debug(f"parse method: {parse_method}.")

        parsed_text = self.parse(
            input_file=input_file,
            parser=parser,
            parse_method=parse_method,
            save_images=save_images,
            save_metadata=save_metadata,
            **kwargs,
        )

        if check_language and not self.check_text_is_english(parsed_text):
            bad_english_error = f"{input_file} was not parsed with good English."
            raise BadEnglishError(bad_english_error)

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            if Path(out_path).is_file():


            with Path(output_file).open("w") as outfile:
                outfile.write(parsed_text)
                return str(output_file)
        return parsed_text

    def parse(
        self,
        input_file: str | PathLike,
        parser: ParserLibrary,
        parse_method: Callable[[str | PathLike[Any], ParserLibrary], str],
        *,
        return_metadata: bool = False,
        return_images: bool = False,
        **kwargs,
    ) -> str:
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
        if return_metadata and "json" not in parser.output_file_types:
            metadata_not_allowed = (
                f"metadata out not permitted for parser {parser.name}."
            )
            raise InvalidOutputFileTypeError(metadata_not_allowed)
        if return_images and "png" not in parser.output_file_types:
            images_not_allowed = f"images out not permitted for parser {parser.name}."
            raise InvalidOutputFileTypeError(images_not_allowed)

        return parse_method(
            input_file, parser, return_metadata, return_images, **kwargs
        )

    # @staticmethod
    # def write_to_markdown_file(parsed_text: str, output_file: str | PathLike) -> str:
    #     """
    #     Write parsed text to markdown file.

    #     Args:
    #         parsed_text (str): _description_
    #         output_file (str | PathLike): _description_

    #     Returns:
    #         str: _description_

    #     """
    #     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    #     logger.debug(f"writing file {output_file}...")
    #     Path(output_file).write_text(parsed_text, encoding="utf-8")

    #     return str(output_file)

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
    def parse_pdf(
        file: str | PathLike,
        parser: ParserLibrary,
        *,
        return_images: bool = False,
        return_metadata: bool = False,
    ) -> str | tuple[str, Any, Any]:
        """
        Parse pdf file to string in md format.

        Args:
            file (str | PathLike): Path to pdf file.
            parser (ParserLibrary): The Parser to use.

        Raises:
            FileParserMismatchError: If an illegal parser was selected.

        Returns:
            str: The text, as str, formatted as markdown.

        """
        logger.debug(f"parsing file {file} using parser {parser}...")
        if parser == MarkerParser:
            rendered = converter(file)
            text, metadata, images = text_from_rendered(rendered)
            logger.debug("successfully parsed pdf.")
            out = text

        if return_metadata:
            out = out, metadata
        if return_images:
            out = out, images

        return out

    @staticmethod
    def parse_epub(file: str | PathLike, parser: ParserLibrary) -> str:
        """
        Parse epub file to string in md format.

        Args:
            file (str | PathLike): Path to epub file.
            parser (ParserLibrary): The Parser to use.

        Raises:
            FileParserMismatchError: If an illegal parser was selected.

        Returns:
            str: The text, as str, formatted as markdown.

        """
        logger.debug(f"parsing file {file} using parser {parser}...")
        if parser == ParserLibrary.PANDOC:
            logger.debug("successfully parsed epub.")
            return pypandoc.convert_file(file, to="md", format="epub")
        bad_parser_file_combo = f"Unsupported parser {parser} for file {file}."
        raise FileParserMismatchError(bad_parser_file_combo)

    @staticmethod
    def parse_html(file: str | PathLike, parser: ParserLibrary) -> str:
        """
        Parse html file to string in md format.

        Args:
            file (str | PathLike): Path to html file.
            parser (ParserLibrary): The Parser to use.

        Raises:
            FileParserMismatchError: If an illegal parser was selected.

        Returns:
            str: The text, as str, formatted as markdown.

        """
        logger.debug(f"parsing file {file} using parser {parser}...")
        if parser == ParserLibrary.PANDOC:
            logger.debug("successfully parsed html.")
            return pypandoc.convert_file(file, to="md", format="html")
        bad_parser_file_combo = f"Unsupported parser {parser} for file {file}."
        raise FileParserMismatchError(bad_parser_file_combo)

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
