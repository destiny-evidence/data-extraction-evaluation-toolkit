"""Tests for deet/utils/cli_utils.py."""

from unittest.mock import patch

import pytest
import typer

from deet.utils.cli_utils import echo_and_log, fail_with_message, optional_progress


def test_echo_and_log(capsys):
    """Capture log; capture stdout, ensure the message appears there."""
    test_message = "Test message for echo and log"

    with patch("deet.utils.cli_utils.logger") as mock_logger:
        echo_and_log(test_message)

    # check stdout (typer echo/secho) contains the message
    captured = capsys.readouterr()
    assert test_message in captured.out

    # check logger.info was called with the message
    mock_logger.info.assert_called_once()
    assert test_message in mock_logger.info.call_args[0][0]


def test_echo_and_log_formatting(capsys):
    """Ensure formatting for typer.secho passes through via kwargs."""
    test_message = "Formatted message"

    with (
        patch("deet.utils.cli_utils.logger"),
        patch("deet.utils.cli_utils.typer.secho") as mock_secho,
    ):
        echo_and_log(test_message, fg=typer.colors.GREEN, bold=True)

    # ensure typer.secho was called with the formatting kwargs
    mock_secho.assert_called_once_with(test_message, fg=typer.colors.GREEN, bold=True)


def test_fail_with_message(capsys):
    """Ensure message is in stderr; ensure exit code is 1."""
    test_message = "Error message for failure"

    with (
        patch("deet.utils.cli_utils.logger"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        fail_with_message(test_message)

    assert exc_info.value.exit_code == 1

    captured = capsys.readouterr()
    assert test_message in captured.err


def test_optional_progress_with_progress_disabled():
    """Test optional_progress yields iterable unchanged when disabled."""
    items = [1, 2, 3, 4, 5]

    with optional_progress(items, show_progress=False) as result:
        collected = list(result)

    assert collected == items


def test_optional_progress_with_progress_enabled():
    """Test optional_progress wraps iterable in progress bar when enabled."""
    items = [1, 2, 3, 4, 5]

    with patch("deet.utils.cli_utils.typer.progressbar") as mock_progressbar:
        mock_progressbar.return_value.__enter__ = lambda _: iter(items)
        mock_progressbar.return_value.__exit__ = lambda _, *__: None

        with optional_progress(items, show_progress=True, label="Testing") as result:
            collected = list(result)

    mock_progressbar.assert_called_once_with(items, label="Testing")
    assert collected == items


def test_optional_progress_default_label():
    """Test optional_progress uses default label."""
    items = [1, 2, 3]

    with patch("deet.utils.cli_utils.typer.progressbar") as mock_progressbar:
        mock_progressbar.return_value.__enter__ = lambda _: iter(items)
        mock_progressbar.return_value.__exit__ = lambda _, *__: None

        with optional_progress(items, show_progress=True):
            pass

    mock_progressbar.assert_called_once_with(items, label="Processing")
