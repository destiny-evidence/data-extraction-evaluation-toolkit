"""Map EPPI ``AdditionalText`` and codes onto typed annotation ``raw_data`` values."""

import json
from typing import Any, cast

from deet.data_models.base import AttributeType

EppiRawDataValue = bool | str | int | float | list[Any] | dict[str, Any]


def eppi_output_data_from_eppi_fields(  # noqa: PLR0911
    output_data_type: AttributeType,
    *,
    additional_text: str,
) -> EppiRawDataValue:
    """
    Map EPPI ``Codes`` row evidence onto typed ``raw_data`` for coerced ``output_data``.

    A row in ``References[].Codes`` means the reviewer applied that code (e.g. ticked
    the checkbox). For boolean attributes, that application is ``True`` even when
    ``AdditionalText`` is empty.

    For every non-boolean type, only the info-box ``AdditionalText`` is used.
    ``ItemAttributeFullTextDetails`` is not used for the stored value (it may still be
    attached to the model for other uses).

    Args:
        output_data_type: Target attribute type (from codeset or prompt CSV).
        additional_text: EPPI ``AdditionalText`` / info-box value.

    Returns:
        Value to store in ``GoldStandardAnnotation.raw_data`` (then coerced via
        ``output_data``).

    """
    additional = (additional_text or "").strip()

    if output_data_type == AttributeType.BOOL:
        return True
    if output_data_type == AttributeType.STRING:
        return additional

    if output_data_type == AttributeType.INTEGER:
        if not additional:
            return output_data_type.missing_annotation_default()
        try:
            return int(float(additional))
        except ValueError:
            return output_data_type.missing_annotation_default()

    if output_data_type == AttributeType.FLOAT:
        if not additional:
            return output_data_type.missing_annotation_default()
        try:
            return float(additional.replace(",", ""))
        except ValueError:
            return output_data_type.missing_annotation_default()

    if output_data_type in (AttributeType.LIST, AttributeType.DICT):
        if not additional:
            return output_data_type.missing_annotation_default()
        try:
            parsed: Any = json.loads(additional)
        except (json.JSONDecodeError, TypeError):
            return output_data_type.missing_annotation_default()
        py_type = output_data_type.to_python_type()
        if isinstance(parsed, py_type):
            return cast("EppiRawDataValue", parsed)
        return output_data_type.missing_annotation_default()

    unsupported = f"Unsupported AttributeType for EPPI mapping: {output_data_type}"
    raise ValueError(unsupported)
