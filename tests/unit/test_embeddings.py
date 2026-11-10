"""Unit tests for the optional fastembed encoder."""

import sys

import numpy as np
import pytest

from deet.evaluators.embeddings import (
    _l2_normalize,
    _load_text_embedding_factory,
    clear_embedding_model_cache,
    encode_texts,
    get_embedding_model,
)


def test_l2_normalize_makes_unit_rows() -> None:
    """Each embedding row has length 1 after normalisation."""
    vectors = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    normalised = _l2_normalize(vectors)
    norms = np.linalg.norm(normalised, axis=1)
    assert norms == pytest.approx([1.0, 1.0])
    assert normalised[0, 0] == pytest.approx(1.0)
    assert normalised[1, 1] == pytest.approx(1.0)


def test_missing_semantic_extra_raises_install_message(monkeypatch) -> None:
    """Loading the encoder without fastembed explains how to install it."""
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ImportError, match="uv sync --extra semantic"):
        _load_text_embedding_factory()


@pytest.mark.no_embedding_mock
def test_encode_texts_empty_input_skips_the_model() -> None:
    """An empty batch does not load fastembed."""
    encoded = encode_texts([])
    assert encoded.shape == (0, 0)


@pytest.mark.no_embedding_mock
def test_encode_texts_returns_unit_vectors(monkeypatch) -> None:
    """encode_texts stacks fastembed rows and L2-normalises them."""

    class _FakeModel:
        def embed(self, documents: list[str]) -> list[np.ndarray]:
            assert documents == ["alpha"]
            return [np.array([3.0, 0.0], dtype=np.float64)]

    def _fake_model(*_args: object, **_kwargs: object) -> _FakeModel:
        return _FakeModel()

    monkeypatch.setattr(
        "deet.evaluators.embeddings.get_embedding_model",
        _fake_model,
    )
    encoded = encode_texts(["alpha"], model_name="unit-test-model")
    assert encoded.shape == (1, 2)
    assert encoded[0, 0] == pytest.approx(1.0)
    assert encoded[0, 1] == pytest.approx(0.0)


@pytest.mark.no_embedding_mock
def test_get_embedding_model_caches_by_name(monkeypatch) -> None:
    """The same model id is constructed once."""
    clear_embedding_model_cache()
    created: list[str] = []

    class _FakeModel:
        def embed(self, documents: list[str]) -> list[np.ndarray]:
            return []

    def _factory(model_name: str) -> _FakeModel:
        created.append(model_name)
        return _FakeModel()

    monkeypatch.setattr(
        "deet.evaluators.embeddings._load_text_embedding_factory",
        lambda: _factory,
    )
    try:
        first = get_embedding_model("unit-test-model")
        second = get_embedding_model("unit-test-model")
        assert first is second
        assert created == ["unit-test-model"]
    finally:
        clear_embedding_model_cache()
