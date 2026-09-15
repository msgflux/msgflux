"""Tests for gateway session/cache header helpers."""

from msgflux.models.session import (
    USER_AGENT,
    merge_session_headers,
    session_extra_headers,
)
from msgflux.runtime.context import thread_context


def test_user_agent_always_present():
    assert session_extra_headers("x-grok-conv-id") == {"User-Agent": USER_AGENT}


def test_session_header_uses_active_thread():
    with thread_context(thread_id="thread_1"):
        assert session_extra_headers("x-grok-conv-id") == {
            "User-Agent": USER_AGENT,
            "x-grok-conv-id": "thread_1",
        }


def test_merge_preserves_caller_headers():
    with thread_context(thread_id="thread_1"):
        params = merge_session_headers(
            {"extra_headers": {"User-Agent": "custom", "X-Other": "1"}},
            "x-opencode-session",
        )

    assert params["extra_headers"] == {
        "User-Agent": "custom",
        "X-Other": "1",
        "x-opencode-session": "thread_1",
    }


def test_merge_without_active_thread_sends_only_user_agent():
    params = merge_session_headers({}, "x-opencode-session")

    assert params["extra_headers"] == {"User-Agent": USER_AGENT}
