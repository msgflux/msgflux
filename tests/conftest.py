"""Shared fixtures for msgflux tests."""

from typing import Any, Dict

import pytest

from msgflux.data.stores import Store


@pytest.fixture(params=["in_memory", "sqlite"])
def checkpoint_store(request, tmp_path):
    """Shared atomic checkpoint gate; register new adapters here."""
    options = (
        {"path": str(tmp_path / "checkpoints.db")} if request.param == "sqlite" else {}
    )
    store = Store.checkpoint(request.param, **options)
    yield store
    if hasattr(store, "close"):
        store.close()


@pytest.fixture(params=["in_memory", "sqlite"])
def approval_journal(request, tmp_path):
    """Shared journal gate with an explicitly controlled wall clock."""
    clock = [100.0]
    options = (
        {"path": str(tmp_path / "approvals.db")} if request.param == "sqlite" else {}
    )
    store = Store.approval(request.param, clock=lambda: clock[0], **options)
    yield store, clock
    store.close()


@pytest.fixture
def sample_state_dict() -> Dict[str, Any]:
    """Sample state dictionary for testing serialization."""
    return {
        "param1": "value1",
        "param2": 42,
        "param3": ["item1", "item2"],
    }


@pytest.fixture
def mock_model_registry():
    """Mock model registry for testing."""
    from msgflux.models.base import BaseModel

    class MockChatModel(BaseModel):
        model_type = "chat_completion"
        provider = "mock"

        def __init__(self, model_id: str = "test-model", **kwargs):
            self.model_id = model_id
            self._api_key = kwargs.get("api_key")
            self.model = None
            self.processor = None
            self.client = None

        def _initialize(self):
            """Initialize mock model."""
            pass

        def __call__(self, *args, **kwargs):
            """Mock call."""
            return {"response": "mock response"}

    return {"chat_completion": {"mock": MockChatModel}}
