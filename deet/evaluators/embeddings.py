"""Optional fastembed encoder for semantic evaluation metrics."""

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, cast

import numpy as np

DEFAULT_EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

SEMANTIC_EXTRA_INSTALL_MESSAGE: str = (
    "Semantic similarity metrics require the optional extra. "
    "Install it with `uv sync --extra semantic` or "
    "`pip install '.[semantic]'`, or leave "
    "`enable_semantic_metrics` false in the extraction config YAML."
)


class TextEmbeddingLike(Protocol):
    """Minimal embed interface used from fastembed."""

    def embed(self, documents: Sequence[str]) -> Iterator[np.ndarray]:
        """Embed documents to one vector per string."""
        ...


_MODEL_CACHE: dict[str, TextEmbeddingLike] = {}


def _load_text_embedding_factory() -> Callable[[str], TextEmbeddingLike]:
    """
    Import fastembed's ``TextEmbedding``, or explain how to install it.

    Returns:
        The ``TextEmbedding`` constructor.

    Raises:
        ImportError: If the ``semantic`` extra is not installed.

    """
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise ImportError(SEMANTIC_EXTRA_INSTALL_MESSAGE) from exc
    return cast("Callable[[str], TextEmbeddingLike]", TextEmbedding)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Scale each row to unit length.

    A dot product of two unit vectors is their cosine similarity.

    Args:
        vectors: Array of shape ``(n, dim)``.

    Returns:
        Row-normalised array. Empty input is returned unchanged.

    """
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    return vectors / safe_norms


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> TextEmbeddingLike:
    """
    Return a cached fastembed model for ``model_name``.

    Args:
        model_name: fastembed model id.

    Returns:
        A model exposing ``embed``.

    Raises:
        ImportError: If the ``semantic`` extra is not installed.

    """
    if model_name not in _MODEL_CACHE:
        factory = _load_text_embedding_factory()
        _MODEL_CACHE[model_name] = factory(model_name)
    return _MODEL_CACHE[model_name]


def encode_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> np.ndarray:
    """
    Encode strings to L2-normalised embedding vectors.

    Args:
        texts: Strings to embed (already normalised by the caller).
        model_name: fastembed model id.

    Returns:
        Array of shape ``(len(texts), embedding_dim)``. Empty input yields
        shape ``(0, 0)``.

    Raises:
        ImportError: If the ``semantic`` extra is not installed.

    """
    if not texts:
        return np.empty((0, 0), dtype=np.float64)
    model = get_embedding_model(model_name)
    rows = [np.asarray(vector, dtype=np.float64) for vector in model.embed(list(texts))]
    embeddings = np.stack(rows, axis=0)
    return _l2_normalize(embeddings)


def clear_embedding_model_cache() -> None:
    """Drop cached embedding models (used in tests)."""
    _MODEL_CACHE.clear()
