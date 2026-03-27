"""Tests for msgflux.nn.modules.embedder module."""

import pytest
from unittest.mock import Mock, patch

from msgflux.nn.modules.embedder import Embedder
from msgflux.core.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockEmbedderModel(BaseModel):
    """Mock embedder model for testing."""

    model_type = "text_embedder"
    provider = "mock"
    batch_support = True

    def __init__(self, batch_support=True):
        self.model_id = "test-embedder"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None
        self.batch_support = batch_support
        self.model_type = "text_embedder"

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        data = kwargs.get("data", [])
        # Return list of embeddings
        if isinstance(data, list):
            embeddings = [[0.1, 0.2, 0.3] for _ in data]
        else:
            embeddings = [[0.1, 0.2, 0.3]]

        response = ModelResponse()
        response.set_response_type("embeddings")
        response.add(embeddings)
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


class TestEmbedder:
    """Test suite for Embedder module."""

    def test_embedder_initialization(self):
        """Test Embedder basic initialization."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        assert embedder.model is mock_model

    def test_embedder_initialization_with_config(self):
        """Test Embedder initialization with configuration."""
        mock_model = MockEmbedderModel()
        config = {"dimensions": 1536, "encoding_format": "float"}
        embedder = Embedder(model=mock_model, config=config)

        assert embedder._buffers["config"]["dimensions"] == 1536

    def test_embedder_inheritance_from_module(self):
        """Test that Embedder inherits from Module."""
        from msgflux.nn.modules.module import Module

        assert issubclass(Embedder, Module)

    def test_embedder_with_pooling_strategy(self):
        """Test Embedder with pooling strategy."""
        mock_model = Mock()
        mock_model.model_type = "audio_embedder"
        config = {"pooling_strategy": "mean"}
        embedder = Embedder(model=mock_model, config=config)

        assert embedder._buffers["config"]["pooling_strategy"] == "mean"

    def test_embedder_invalid_model_type(self):
        """Test Embedder raises TypeError for invalid model type."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        with pytest.raises(TypeError, match="requires be `embedder` model"):
            Embedder(model=mock_model)

    def test_embedder_forward_single_string(self):
        """Test Embedder forward with single string."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        result = embedder("Hello world")

        assert isinstance(result, list)
        assert len(result) == 3  # Single embedding vector
        assert result == [0.1, 0.2, 0.3]

    def test_embedder_forward_list_of_strings(self):
        """Test Embedder forward with list of strings."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        result = embedder(["Hello", "World"])

        assert isinstance(result, list)
        assert len(result) == 2  # Two embeddings
        assert all(isinstance(emb, list) for emb in result)

    def test_embedder_forward_with_message(self):
        """Test Embedder forward with Message object."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model, message_fields={"task": "text"})

        msg = Message()
        msg.text = "Hello world"
        result = embedder(msg)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embedder_aforward_single_string(self):
        """Test Embedder async forward with single string."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        result = await embedder.aforward("Hello world")

        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embedder_aforward_list_of_strings(self):
        """Test Embedder async forward with list of strings."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        result = await embedder.aforward(["Hello", "World"])

        assert isinstance(result, list)
        assert len(result) == 2

    def test_embedder_batch_support_true(self):
        """Test Embedder with batch_support=True."""
        mock_model = MockEmbedderModel(batch_support=True)
        embedder = Embedder(model=mock_model)

        result = embedder(["Text 1", "Text 2", "Text 3"])

        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)

    def test_embedder_batch_support_false(self):
        """Test Embedder with batch_support=False uses map_gather."""
        mock_model = MockEmbedderModel(batch_support=False)
        embedder = Embedder(model=mock_model)

        # This should use F.map_gather internally
        with patch("msgflux.nn.functional.map_gather") as mock_map_gather:
            # Mock map_gather to return list of responses
            mock_responses = [
                MockEmbedderModel()(data=["Text 1"]),
                MockEmbedderModel()(data=["Text 2"]),
            ]
            mock_map_gather.return_value = mock_responses

            result = embedder(["Text 1", "Text 2"])

            # Verify map_gather was called
            assert mock_map_gather.called

    def test_embedder_config_invalid_type(self):
        """Test Embedder raises TypeError for invalid config type."""
        mock_model = MockEmbedderModel()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            Embedder(model=mock_model, config="invalid")

    def test_embedder_inspect_model_execution_params(self):
        """Test inspect_model_execution_params method."""
        mock_model = MockEmbedderModel()
        config = {"normalize": True, "truncate": False}
        embedder = Embedder(model=mock_model, config=config)

        params = embedder.inspect_model_execution_params("Test text")

        assert params["data"] == "Test text"
        assert params["normalize"] is True
        assert params["truncate"] is False

    def test_embedder_with_response_mode(self):
        """Test Embedder with custom response_mode."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model, response_mode="embeddings.output")

        assert embedder.response_mode == "embeddings.output"

    def test_embedder_text_type(self):
        """Test text_embedder model type."""
        mock_model = Mock()
        mock_model.model_type = "text_embedder"
        embedder = Embedder(model=mock_model)

        assert "embedder" in embedder.model.model_type

    def test_embedder_image_type(self):
        """Test image_embedder model type."""
        mock_model = Mock()
        mock_model.model_type = "image_embedder"
        embedder = Embedder(model=mock_model)

        assert "embedder" in embedder.model.model_type

    def test_embedder_audio_type(self):
        """Test audio_embedder model type."""
        mock_model = Mock()
        mock_model.model_type = "audio_embedder"
        embedder = Embedder(model=mock_model)

        assert "embedder" in embedder.model.model_type

    def test_embedder_empty_list(self):
        """Test Embedder with empty list."""
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)

        result = embedder([])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_embedder_with_normalize_config(self):
        """Test Embedder with normalize configuration."""
        mock_model = MockEmbedderModel()
        config = {"normalize": True}
        embedder = Embedder(model=mock_model, config=config)

        params = embedder.inspect_model_execution_params("Test")
        assert params.get("normalize") is True

    def test_embedder_with_truncate_config(self):
        """Test Embedder with truncate configuration."""
        mock_model = MockEmbedderModel()
        config = {"truncate": True}
        embedder = Embedder(model=mock_model, config=config)

        params = embedder.inspect_model_execution_params("Test")
        assert params.get("truncate") is True
