from unittest.mock import patch

import pytest

from deet.hierarchical_mvp.utils import configure_lm, get_api_base_for_model


@pytest.mark.parametrize(
    ("model", "expected_api_base"),
    [
        ("azure/gpt-5.2", "https://openai.example.test"),
        ("anthropic/claude-sonnet", "https://anthropic.example.test"),
    ],
)
def test_get_api_base_for_model(monkeypatch, model, expected_api_base):
    monkeypatch.setenv("AZURE_API_BASE", "https://openai.example.test")
    monkeypatch.setenv(
        "AZURE_API_BASE_ANTHROPIC", "https://anthropic.example.test"
    )

    assert get_api_base_for_model(model) == expected_api_base


def test_get_api_base_for_anthropic_model_requires_anthropic_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_API_BASE", "https://openai.example.test")
    monkeypatch.delenv("AZURE_API_BASE_ANTHROPIC", raising=False)

    with pytest.raises(OSError, match="AZURE_API_BASE_ANTHROPIC is not set"):
        get_api_base_for_model("anthropic/claude-sonnet")


def test_configure_lm_uses_anthropic_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://openai.example.test")
    monkeypatch.setenv(
        "AZURE_API_BASE_ANTHROPIC", "https://anthropic.example.test"
    )

    with (
        patch("deet.hierarchical_mvp.utils.dspy.LM") as lm_class,
        patch("deet.hierarchical_mvp.utils.dspy.configure") as configure,
    ):
        configure_lm("anthropic/claude-sonnet", 4096, cache=True)

    lm_class.assert_called_once_with(
        model="anthropic/claude-sonnet",
        api_key="test-key",
        api_base="https://anthropic.example.test",
        max_tokens=4096,
        cache=True,
    )
    configure.assert_called_once_with(lm=lm_class.return_value)
