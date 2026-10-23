"""Integration tests for dotdict persistent store support."""

import pytest

from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message


def test_dotdict_persistent_store_with_diskcache(tmp_path):
    """Validate dotdict persistence against a real diskcache store."""
    diskcache = pytest.importorskip("diskcache")
    cache = diskcache.Cache(str(tmp_path / "dotdict-cache"))

    try:
        message = dotdict({"status": "ready"}, store=cache, store_prefix="run_1")
        message.set("user.name", "Maria")
        message.update({"metrics.latency_ms": 42})
        message.config = {"debug": False}

        other = dotdict(store=cache, store_prefix="run_2")
        other.user = {"name": "Ana"}

        restored = dotdict(store=cache, store_prefix="run_1")

        assert restored.status == "ready"
        assert restored.user.name == "Maria"
        assert restored.metrics.latency_ms == 42
        assert restored.config.debug is False
        assert cache["run_1.user"] == {"name": "Maria"}
        assert cache["run_1.metrics"] == {"latency_ms": 42}

        restored.user.name = "Ana"
        assert dotdict(store=cache, store_prefix="run_1").user.name == "Maria"

        restored.user = {**restored.user, "name": "Ana"}
        assert dotdict(store=cache, store_prefix="run_1").user.name == "Ana"
        assert dotdict(store=cache, store_prefix="run_2").user.name == "Ana"

        del restored.status
        assert "run_1.status" not in cache
    finally:
        cache.close()


def test_message_persistent_store_with_diskcache(tmp_path):
    """Validate Message persistence against a real diskcache store."""
    diskcache = pytest.importorskip("diskcache")
    cache = diskcache.Cache(str(tmp_path / "message-cache"))

    try:
        message = Message(
            content="Analyze this",
            context={"request_id": "req-1"},
            store=cache,
            store_prefix="msg_1",
        )
        message.set("outputs.answer", "done")
        message.update({"extra.tokens": 12})

        restored = Message(store=cache, store_prefix="msg_1")

        assert restored.content == "Analyze this"
        assert restored.context.request_id == "req-1"
        assert restored.outputs.answer == "done"
        assert restored.extra.tokens == 12
        assert restored.response == {}

        restored.outputs.answer = "changed"
        assert Message(store=cache, store_prefix="msg_1").outputs.answer == "done"

        restored.outputs = {**restored.outputs, "answer": "changed"}
        assert Message(store=cache, store_prefix="msg_1").outputs.answer == "changed"
    finally:
        cache.close()
