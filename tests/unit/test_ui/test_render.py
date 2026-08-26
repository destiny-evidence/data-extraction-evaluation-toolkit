"""Tests for deet.ui.terminal.render."""

from unittest.mock import patch

from deet.ui.terminal.render import optional_progress


def test_optional_progress_with_progress_disabled():
    """Test optional_progress yields iterable unchanged when disabled."""
    items = [1, 2, 3, 4, 5]

    with optional_progress(items, show_progress=False) as result:
        collected = list(result)

    assert collected == items


def test_optional_progress_with_progress_enabled():
    """Test optional_progress uses default label."""
    items = [1, 2, 3]
    label = "Testing"

    with patch("deet.ui.terminal.render.Progress") as mock_progress:
        mock_instance = mock_progress.return_value
        mock_instance.__enter__.return_value = mock_instance

        mock_instance.add_task.return_value = 1

        with optional_progress(items, show_progress=True, label=label) as result:
            collected = list(result)

        mock_instance.add_task.assert_called_once_with(description=label, total=3)

        assert mock_instance.advance.call_count == len(items)
        assert collected == items
