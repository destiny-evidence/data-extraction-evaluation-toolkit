"""Tools for text quality assessments."""

import re
from enum import Enum

from diskcache import Cache

from deet.settings import get_settings

# CACHE init
CACHE_DIR = get_settings().base_disk_cache_dir / "nltk_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
nltk_cache = Cache(str(CACHE_DIR))


class Language(Enum):
    """Enum of languages for quality-checking."""

    ENGLISH = "en"


@nltk_cache.memoize(typed=True, expire=None, tag="nltk-brown-words")
def get_bc_brown_words() -> set:
    """Get bc_brown words, cached."""
    from nltk.corpus import brown
    from nltk.data import find

    try:
        find("corpora/brown")
    except LookupError:
        from nltk import download

        download("brown")

    return {x.lower() for x in brown.words()}


# NOTE: no longer eagerly computed at import time - only when actually needed,
# and only if `nltk` is installed / feature is enabled.
_bc_brown_cache: set | None = None


def _get_bc_brown() -> set:
    """Lazily load/cache brown words, importing nltk only on first use."""
    global _bc_brown_cache  # noqa: PLW0603
    if _bc_brown_cache is None:
        _bc_brown_cache = get_bc_brown_words()
    return _bc_brown_cache


class EmptyTextError(Exception):
    """
    Raise when our text passed to is_english is empty.

    Args:
        Exception (_type_):

    """

    def __init__(self, msg: str = "Supplied text is empty.", *args, **kwargs) -> None:  # noqa: ANN002
        """Init the exception with default message."""
        super().__init__(msg, *args, **kwargs)


def check_language(
    text: str, lang: Language = Language.ENGLISH, threshold: float = 0.2
) -> bool:
    """
    Assess if text is in the specified language.

    Args:
        text (str): Text to assess.
        lang (Language): Language to check against.
        threshold (float): Threshold for word overlap.

    Raises:
        EmptyTextError: If text is empty.

    Returns:
        bool: True if text matches language, False otherwise.

    """
    if text is None or text == "":
        raise EmptyTextError
    tok = set(re.findall(r"\w+", text.lower()))

    if len(tok) <= 0:
        raise EmptyTextError

    if isinstance(lang, str):
        lang = Language(lang)

    if lang == Language.ENGLISH:
        bc_brown = _get_bc_brown()
        return len(tok & bc_brown) / len(tok) > threshold
    missing_lang = f"Language '{lang.value}' not supported yet."
    raise NotImplementedError(missing_lang)


#  backward compatibility
def is_english(text: str, threshold: float = 0.2) -> bool:
    """
    Check if text meets minimum English quality.

    Args:
        text (str): the text to check.
        threshold (float, optional): Defaults to 0.2.

    Returns:
        bool: True if 'English', false if not.

    """
    return check_language(text, lang=Language.ENGLISH, threshold=threshold)
