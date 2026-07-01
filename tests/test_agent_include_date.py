"""Tests for Agent include_date feature with weekday."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from msgflux.nn.modules import Agent


def test_agent_include_date_with_weekday():
    """Test that include_date includes the day of the week."""

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create agent with include_date enabled
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_message="You are a helpful assistant",
        config={"include_date": True},
    )

    # Mock datetime to have a predictable date
    mock_datetime = datetime(2025, 12, 9, 10, 30, 0, tzinfo=timezone.utc)  # Tuesday

    with patch("msgflux.utils.time.datetime") as mock_dt:
        mock_dt.now.return_value = mock_datetime
        mock_dt.strftime = datetime.strftime  # Keep strftime working

        # Get the system prompt
        system_prompt = agent.get_system_prompt()

        # Verify that the date includes the weekday
        assert "Tuesday" in system_prompt, "Weekday should be included"
        assert "December" in system_prompt, "Month name should be included"
        assert "09" in system_prompt, "Day should be included"
        assert "2025" in system_prompt, "Year should be included"

        # Verify the full format
        assert "Tuesday, December 09, 2025" in system_prompt


def test_agent_without_include_date():
    """Test that date is not included when include_date is False."""

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create agent without include_date
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_message="You are a helpful assistant",
        config={"include_date": False},
    )

    # Get the system prompt
    system_prompt = agent.get_system_prompt()

    # Verify that date-related text is not present
    assert "current date" not in system_prompt.lower()


def test_agent_include_date_default_false():
    """Test that include_date defaults to False."""

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create agent without specifying include_date
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_message="You are a helpful assistant",
    )

    # Get the system prompt
    system_prompt = agent.get_system_prompt()

    # Verify that date is not included by default
    assert "current date" not in system_prompt.lower()


def test_agent_include_date_format_consistency():
    """Test that the date format is consistent across different dates."""

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create agent with include_date
    agent = Agent(
        name="test_agent",
        model=mock_model,
        system_message="You are a helpful assistant",
        config={"include_date": True},
    )

    # Test multiple dates
    test_dates = [
        (
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "Wednesday, January 01, 2025",
        ),
        (
            datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            "Wednesday, December 31, 2025",
        ),
        (datetime(2025, 7, 4, 12, 0, 0, tzinfo=timezone.utc), "Friday, July 04, 2025"),
    ]

    for mock_datetime, expected_date in test_dates:
        with patch("msgflux.utils.time.datetime") as mock_dt:
            mock_dt.now.return_value = mock_datetime

            # Get the system prompt
            system_prompt = agent.get_system_prompt()

            # Verify the expected date format
            assert expected_date in system_prompt, (
                f"Expected date '{expected_date}' not found in prompt"
            )
