"""Text utilities."""

import re


def slugify(value: str) -> str:
    """Turn a string into a filesystem-safe slug (lowercase, hyphen-separated)."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        invalid = f"Could not derive a slug from {value!r}"
        raise ValueError(invalid)
    return slug
