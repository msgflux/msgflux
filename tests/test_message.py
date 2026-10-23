"""Tests for Message defaults and hidden key behavior."""

from msgflux.core.message import Message


def test_message_hidden_keys_hide_default_field_but_keep_direct_access():
    """Hidden Message fields stay accessible via direct access."""
    msg = Message(extra={"token": "secret"}, hidden_keys=["extra"])

    assert "extra" not in msg
    assert msg.extra.token == "secret"
    assert msg.to_dict()["content"] is None
    assert "extra" not in msg.to_dict()


def test_message_store_hydration_preserves_hidden_field():
    """Hydrated hidden fields are not overwritten by Message defaults."""
    store = {"chat_1.extra": {"token": "secret"}}

    msg = Message(store=store, store_prefix="chat_1", hidden_keys=["extra"])

    assert "extra" not in msg
    assert msg.extra.token == "secret"
    assert "extra" not in msg.to_dict()
    assert msg.outputs == {}
    assert msg.response == {}
