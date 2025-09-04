"""Utilities for parsing input files (e.g. pdf) for documents into output files (e.g. md)."""

from enum import StrEnum, auto
from os import PathLike

import pypandoc
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output, text_from_rendered


class InvalidInputFileTypeError(Exception):
    """
    Raise when user supplies a not permitted input file type.

    Args:
        Exception (_type_):

    """


class InputFileType(StrEnum):
    """
    Enumeration of permitted input file types.

    Args:
        StrEnum (_type_):

    """

    PDF = auto()
    EPUB = auto()
    HTML = auto()
    JPEG = auto()
    PNG = auto()


class ParserLibrary(StrEnum):
    """
    Enumeration of permitted (external) parser libraries.

    Args:
        StrEnum (_type_):

    """

    MARKER = auto()
    PANDOC = auto()


class DocumentParser:
    """Parse documents from target format to other target format."""

    def __init__(
        self,
        default_parser_pdf: ParserLibrary = ParserLibrary.MARKER,
        default_parser_epub: ParserLibrary = ParserLibrary.PANDOC,
        default_parser_html: ParserLibrary = ParserLibrary.PANDOC,
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

    def __call__(
        self,
        input_file: str | PathLike,
        output_file: str | PathLike | None,
        parser: ParserLibrary | None = None,
        input_file_type: InputFileType | None = None,
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

        Returns:
            str: _description_

        """
        return "test"

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
        return InputFileType(extension)

    @staticmethod
    def parse_pdf(file: str | PathLike, parser: ParserLibrary):
        pass

    @staticmethod
    def parse_epub(file: str | PathLike, parser: ParserLibrary):
        pass

    @staticmethod
    def parse_html(file: str | PathLike, parser: ParserLibrary):
        pass
