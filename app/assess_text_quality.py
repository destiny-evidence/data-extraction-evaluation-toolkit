"""Naive Function for inspecting the quality of parsed text."""

import re

from nltk import download
from nltk.corpus import brown
from nltk.data import find

try:
    find("corpora/brown")
except LookupError:
    download("brown")
bc_brown = {x.lower() for x in brown.words()}


class EmptyTextError(Exception):
    """
    Raise when our text passed to is_english is empty.

    Args:
        Exception (_type_):

    """

    def __init__(self, msg: str = "Supplied text is empty.", *args, **kwargs) -> None:  # noqa: ANN002
        """Init the exception with default message."""
        super().__init__(msg, *args, **kwargs)


def is_english(text: str, threshold: float = 0.2) -> bool:
    """
    Assess if text is English.

    Args:
        text (str): _description_
        threshold (float, optional): _description_. Defaults to 0.2.

    Raises:
        EmptyTextError: _description_

    Returns:
        bool: True if english (above threshold), False otherwise.

    """
    if text is None or text == "":
        raise EmptyTextError
    tok = set(re.findall(r"\w+", text.lower()))

    if len(tok) <= 0:
        raise EmptyTextError

    return len(tok & bc_brown) / len(tok) > threshold
