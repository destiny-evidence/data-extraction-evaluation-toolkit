"""Optional integration test that loads a real fastembed model."""

import pytest

from deet.evaluators.embeddings import clear_embedding_model_cache
from deet.evaluators.metrics import semantic_match_rate


@pytest.mark.slow
def test_semantic_match_rate_number_word_paraphrase() -> None:
    """
    The default fastembed model treats a number-word paraphrase as a match.

    This downloads ``BAAI/bge-small-en-v1.5`` on first run. CI skips tests
    marked ``slow``; run locally with ``pytest -m slow``.
    """
    pytest.importorskip("fastembed")
    clear_embedding_model_cache()
    paraphrase = semantic_match_rate(
        ["three blind mice"],
        ["3 blind mice"],
    )
    unrelated = semantic_match_rate(
        ["hypertension"],
        ["completely different outcome"],
    )
    assert paraphrase == 1.0
    assert unrelated == 0.0
