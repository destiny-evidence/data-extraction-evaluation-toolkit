from unittest.mock import patch

import pytest
import typer

from deet.ui.messenger import fail_with_message, notify


def test_notify(capsys):
    """Capture log; capture stdout, ensure the message appears there *once*."""
    test_message = "Test message for echo and log"

    with patch("deet.ui.messenger.logger") as mock_logger:
        mock_logger.bind.return_value = mock_logger
        notify(test_message)

    # check stdout (typer echo/secho) contains the message once
    captured = capsys.readouterr()
    occurences = captured.out.count(test_message)
    assert occurences == 1

    # check logger.info was called with the message
    mock_logger.log.assert_called_once_with("INFO", test_message)


def test_fail_with_message(capsys):
    """Ensure message is in stderr; ensure exit code is 1."""
    test_message = "Error message for failure"

    with (
        patch("deet.ui.messenger.logger"),
        pytest.raises(typer.Exit) as exc_info,
    ):
        fail_with_message(test_message)

    assert exc_info.value.exit_code == 1

    captured = capsys.readouterr()
    assert test_message in captured.err
