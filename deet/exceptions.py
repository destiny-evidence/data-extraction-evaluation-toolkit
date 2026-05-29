"""Custom exceptions."""

from deet.data_models.base import AttributeType


class MissingDocumentError(Exception):
    """
    Raise when looking up a document by id fails.

    Args:
        Exception (_type_):

    """


class DuplicateAnnotationError(Exception):
    """
    Raise when multiple annotations for a single attribute are present.

    Args:
        Exception (_type_):

    """


class InvalidInputFileTypeError(Exception):
    """
    Raise when user supplies a not permitted input file type.

    Args:
        Exception (_type_):

    """


class InvalidOutputFileTypeError(Exception):
    """
    Raise when user supplies a not permitted output file.

    Args:
        Exception (_type_):

    """


class InvalidFileTypeError(Exception):
    """
    Raise when user supplies a not permitted file.

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


class NoAbstractError(Exception):
    """
    Raise when we can't find an abstract in our citation info.

    Args:
        Exception (_type_): _description_

    """


class UnsupportedEppiAttributeTypeError(ValueError):
    """
    Raised when EPPI ``AdditionalText`` / Codes mapping cannot handle an attribute type.

    Subclasses :class:`ValueError` so existing ``except ValueError`` call sites remain
    valid while allowing targeted handling via ``UnsupportedEppiAttributeTypeError``.

    Args:
        output_data_type: The :class:`~deet.data_models.base.AttributeType` that has
            no mapping in the EPPI ``raw_data`` pipeline (see
            ``deet.processors.eppi_annotation_converter.eppi_output_data_from_eppi_fields``).

    """

    def __init__(
        self,
        output_data_type: AttributeType,
        *,
        message: str | None = None,
    ) -> None:
        """Store ``output_data_type`` and build a default message when omitted."""
        self.output_data_type = output_data_type
        text = message or (
            f"Unsupported AttributeType for EPPI mapping: {output_data_type!s}"
        )
        super().__init__(text)


class LitellmModelNotMappedError(Exception):
    """
    Raised when litellm reports the model is missing from its registry.

    ``litellm.get_max_tokens`` can raise a bare ``Exception`` with a
    characteristic message; we translate that to this type so callers can
    handle it without a broad ``except Exception``.
    """
