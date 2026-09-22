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


@pytest.mark.parametrize(
    ("openai_base", "expected_extra_kwargs"),
    [
        (None, {}),
        (
            "https://openai.example.test/v1",
            {"api_base": "https://openai.example.test/v1"},
        ),
    ],
)
def test_configure_lm_uses_openai_credentials(
    monkeypatch, openai_base, expected_extra_kwargs
):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.delenv("OPENAI_BASE", raising=False)
    if openai_base:
        monkeypatch.setenv("OPENAI_BASE", openai_base)

    with (
        patch("deet.hierarchical_mvp.utils.dspy.LM") as lm_class,
        patch("deet.hierarchical_mvp.utils.dspy.configure") as configure,
    ):
        configure_lm("openai/gpt-5", 4096, cache=True)

    lm_class.assert_called_once_with(
        model="openai/gpt-5",
        api_key="openai-test-key",
        max_tokens=4096,
        cache=True,
        **expected_extra_kwargs,
    )
    configure.assert_called_once_with(lm=lm_class.return_value)
