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


class InvalidFileTypeError(Exception):
    """
    Riase when user supplies a not permitted file.

    Args:
        Exception (_type_):

    """


class FileParserMismatchError(Exception):
    """
    Raise when we have an input-file <> parser mismatch.

    Args:
        Exception (_type_):

    """


class MalformedLanguageError(Exception):
    """
    Raise when language checker fails.

    Args:
        Exception (_type_): _description_

    """


class EmptyPdfExtractionError(Exception):
    """
    Raise when PDF parsing yields no extractable text.

    Occurs when the PDF has no mappable text (e.g. image-only) or when text
    is represented in a way pdfminer cannot decode.

    """

    DEFAULT_MESSAGE = (
        "PDF contained no extractable text (e.g. image-only or text in "
        "unsupported encoding)."
    )


class MissingCitationElementError(Exception):
    """
    Raise when required element of citation is missing.

    Args:
        Exception (_type_): _description_

    """


class BadDocumentIdError(Exception):
    """
    Raise when our `Document.document_id` field
    doesn't satisfy our criteria.

    Args:
        Exception (_type_): _description_

    """


class JsonStyleError(Exception):
    """
    Raise when a json containing document-reference-linkages
    is incorrectly formatted.

    Args:
        Exception (_type_): _description_

    """
