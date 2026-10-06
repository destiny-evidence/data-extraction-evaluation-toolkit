"""Custom exceptions."""

from deet.data_models.base import AttributeType


class KnownError(Exception):
    """
    Base for errors deet recognises and has a user-facing message for.

    Subclass one of the two labels below rather than this class directly.
    Third-party errors are mapped onto these in
    ``deet.extractors.llm_errors.KNOWN_LLM_ERRORS``.
    """


class RunAbortingError(KnownError):
    """
    A known error caused by the setup, so every document would fail the same way.

    The run is stopped and the message is shown to the user.
    """


class DocumentSkippedError(KnownError):
    """
    A known error caused by a single document.

    The document is skipped with a warning and the run continues.
    """


class MissingDocumentError(DocumentSkippedError):
    """
    Raise when looking up a document by id fails.

    Args:
        Exception (_type_):

    """


class DuplicateAnnotationError(DocumentSkippedError):
    """
    Raise when multiple annotations for a single attribute are present.

    Args:
        Exception (_type_):

    """


class InvalidInputFileTypeError(RunAbortingError):
    """
    Raise when user supplies a not permitted input file type.

    Args:
        Exception (_type_):

    """


class InvalidOutputFileTypeError(RunAbortingError):
    """
    Raise when user supplies a not permitted output file.

    Args:
        Exception (_type_):

    """


class InvalidFileTypeError(RunAbortingError):
    """
    Raise when user supplies a not permitted file.

    Args:
        Exception (_type_):

    """


class FileParserMismatchError(RunAbortingError):
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


class EmptyPdfExtractionError(DocumentSkippedError):
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


class JsonStyleError(RunAbortingError):
    """
    Raise when a json containing document-reference-linkages
    is incorrectly formatted.

    Args:
        Exception (_type_): _description_

    """


class NoAbstractError(DocumentSkippedError):
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


class UnsupportedModelParamsError(RunAbortingError):
    """
    Raised when the configured model rejects a configured LLM parameter.

    Wraps ``litellm.UnsupportedParamsError`` (e.g. ``temperature=0.1`` on an
    O-series model) so the run is aborted with an actionable message instead
    of failing silently on every document.
    """


class LLMAuthenticationError(RunAbortingError):
    """Raised when the provider rejects the configured credentials or access."""


class LLMModelNotFoundError(RunAbortingError):
    """Raised when the provider does not know the configured model or deployment."""


class LLMContentFilterError(DocumentSkippedError):
    """Raised when the provider's content filter rejects a document."""


class LLMContextWindowExceededError(DocumentSkippedError):
    """Raised when a document does not fit in the model's context window."""


class LLMRateLimitError(DocumentSkippedError):
    """Raised when the provider rate-limits the request for a document."""


class LLMTimeoutError(DocumentSkippedError):
    """Raised when the request for a document times out."""


class SplitsValidationError(RunAbortingError):
    """Raised when invalid allocation of documents to splits is attemped."""
