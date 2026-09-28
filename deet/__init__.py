"""
DEET package root.

Import anyio eagerly before any module can import dspy. Recent DSPy releases can
install lazy placeholders for anyio/litellm in ``sys.modules``; if FastAPI (via
destiny_sdk.auth) imports anyio submodules while that placeholder is still
resolving, notebooks can hit partially initialized-module errors (e.g.
``anyio.abc`` circular imports or missing ``anyio.lowlevel``).
"""

import anyio  # noqa: F401
import anyio.abc  # noqa: F401
import anyio.lowlevel  # noqa: F401
