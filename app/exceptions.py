"""Custom exceptions."""


class InvalidInputFileTypeError(Exception):
    """
    Raise when user supplies a not permitted input file type.

    Args:
        Exception (_type_):

    """


class InvalidOutputFileTypeError(Exception):
    """
    Riase when user supplies a not permitted output file.

    Args:
        Exception (_type_):

    """


class FileParserMismatchError(Exception):
    """
    Raise when we have an input-file <> parser mismatch.

    Args:
        Exception (_type_):

    """


class BadEnglishError(Exception):
    """
    Raise when our rudimentary English checker fails.

    Args:
        Exception (_type_): _description_

    """
