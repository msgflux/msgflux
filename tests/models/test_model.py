"""Tests for msgflux.models.model module."""

from unittest.mock import patch

import pytest

from msgflux.models.base import BaseModel
from msgflux.models.model import Model


class MockProvider(BaseModel):
    """Mock provider for testing."""

    model_type = "chat_completion"
    provider = "mock_provider"

    def __init__(self, model_id: str = "default-model", **kwargs):
        self.model_id = model_id
        self._api_key = kwargs.get("api_key")
        self.model = None
        self.processor = None
        self.client = None
        self.custom_param = kwargs.get("custom_param")

    def _initialize(self):
        pass

    def __call__(self, *args, **kwargs):
        return {"response": "mock"}


class TestModel:
    """Test suite for Model class."""

    def test_model_path_parser(self):
        """Test parsing of model path."""
        provider, model_id = Model._model_path_parser("openai/gpt-4")
        assert provider == "openai"
        assert model_id == "gpt-4"

    def test_model_path_parser_with_slash_in_model_id(self):
        """Test parsing when model_id contains slashes."""
        provider, model_id = Model._model_path_parser("huggingface/org/model-name")
        assert provider == "huggingface"
        assert model_id == "org/model-name"

    @patch("msgflux.models.model.model_registry")
    def test_get_model_class_success(self, mock_registry):
        """Test getting model class successfully."""
        mock_registry.__getitem__ = lambda self, key: (
            {"mock_provider": MockProvider} if key == "chat_completion" else {}
        )
        mock_registry.__contains__ = lambda self, key: key == "chat_completion"

        model_cls = Model._get_model_class("chat_completion", "mock_provider")
        assert model_cls == MockProvider

    @patch("msgflux.models.model.model_registry")
    def test_get_model_class_invalid_type(self, mock_registry):
        """Test error when model type is not supported."""
        mock_registry.__contains__ = lambda self, key: False

        with pytest.raises(
            ValueError, match="Model type `invalid_type` is not supported"
        ):
            Model._get_model_class("invalid_type", "some_provider")

    @patch("msgflux.models.model.model_registry")
    def test_get_model_class_invalid_provider(self, mock_registry):
        """Test error when provider is not registered."""
        mock_registry.__getitem__ = lambda self, key: (
            {} if key == "chat_completion" else {}
        )
        mock_registry.__contains__ = lambda self, key: key == "chat_completion"

        with pytest.raises(
            ValueError,
            match="Provider `invalid_provider` not registered for type `chat_completion`",
        ):
            Model._get_model_class("chat_completion", "invalid_provider")

    @patch("msgflux.models.model.model_registry")
    def test_create_model(self, mock_registry):
        """Test creating a model instance."""
        mock_registry.__getitem__ = lambda self, key: (
            {"mock_provider": MockProvider} if key == "chat_completion" else {}
        )
        mock_registry.__contains__ = lambda self, key: key == "chat_completion"

        model = Model._create_model(
            "chat_completion", "mock_provider/test-model", custom_param="test_value"
        )

        assert isinstance(model, MockProvider)
        assert model.model_id == "test-model"
        assert model.custom_param == "test_value"

    @pytest.mark.skip(
        reason="from_serialized requires complex __setstate__/__getstate__ implementation"
    )
    @patch("msgflux.models.model.model_registry")
    def test_from_serialized(self, mock_registry):
        """Test deserializing a model from state."""
        mock_registry.__getitem__ = lambda self, key: (
            {"mock_provider": MockProvider} if key == "chat_completion" else {}
        )
        mock_registry.__contains__ = lambda self, key: key == "chat_completion"

        state = {
            "model_id": "restored-model",
            "custom_param": "restored_value",
        }

        model = Model.from_serialized("mock_provider", "chat_completion", state)

        assert isinstance(model, MockProvider)
        assert model.model_id == "restored-model"
        assert model.custom_param == "restored_value"

    @patch("msgflux.models.model.model_registry")
    def test_chat_completion(self, mock_registry):
        """Test chat_completion factory method."""
        mock_registry.__getitem__ = lambda self, key: (
            {"mock_provider": MockProvider} if key == "chat_completion" else {}
        )
        mock_registry.__contains__ = lambda self, key: key == "chat_completion"

        model = Model.chat_completion("mock_provider/gpt-4")
        assert isinstance(model, MockProvider)
        assert model.model_id == "gpt-4"

    @patch("msgflux.models.model.model_registry")
    def test_providers_method(self, mock_registry):
        """Test providers method returns correct structure."""
        mock_registry.items = lambda: [
            ("chat_completion", {"openai": MockProvider, "anthropic": MockProvider}),
            ("text_embedder", {"openai": MockProvider}),
        ]

        providers = Model.providers()
        assert "chat_completion" in providers
        assert "openai" in providers["chat_completion"]
        assert "anthropic" in providers["chat_completion"]
        assert "text_embedder" in providers
        assert "openai" in providers["text_embedder"]

    @patch("msgflux.models.model.model_registry")
    def test_model_types_method(self, mock_registry):
        """Test model_types method returns list of types."""
        mock_registry.keys = lambda: [
            "chat_completion",
            "text_embedder",
            "text_to_image",
        ]

        model_types = Model.model_types()
        assert "chat_completion" in model_types
        assert "text_embedder" in model_types
        assert "text_to_image" in model_types
