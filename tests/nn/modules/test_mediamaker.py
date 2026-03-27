"""Tests for msgflux.nn.modules.mediamaker module."""

import pytest
from unittest.mock import Mock

from msgflux.nn.modules.mediamaker import MediaMaker
from msgflux.core.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockMediaModel(BaseModel):
    """Mock media generation model for testing."""

    model_type = "text_to_image"
    provider = "mock"

    def __init__(self):
        self.model_id = "test-media"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add("generated_media_url")
        return response

    async def acall(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add("generated_media_url")
        return response


class TestMediaMaker:
    """Test suite for MediaMaker module."""

    def test_mediamaker_initialization(self):
        """Test MediaMaker basic initialization."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model)

        assert mediamaker.model is mock_model

    def test_mediamaker_initialization_with_config(self):
        """Test MediaMaker initialization with configuration."""
        mock_model = MockMediaModel()
        config = {"fps": 24, "duration_seconds": 5, "n": 1}
        mediamaker = MediaMaker(model=mock_model, config=config)

        assert mediamaker._buffers["config"]["fps"] == 24
        assert mediamaker._buffers["config"]["duration_seconds"] == 5
        assert mediamaker._buffers["config"]["n"] == 1

    def test_mediamaker_with_negative_prompt(self):
        """Test MediaMaker with negative_prompt."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(
            model=mock_model, negative_prompt="no blur, no artifacts"
        )

        assert mediamaker.negative_prompt == "no blur, no artifacts"

    def test_mediamaker_with_response_format(self):
        """Test MediaMaker with response_format."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model, response_format="base64")

        assert mediamaker.response_format == "base64"

    def test_mediamaker_invalid_response_format_type(self):
        """Test MediaMaker raises TypeError for invalid response_format type."""
        mock_model = MockMediaModel()

        with pytest.raises(TypeError, match="`response_format` need be a str"):
            MediaMaker(model=mock_model, response_format=123)

    def test_mediamaker_invalid_negative_prompt_type(self):
        """Test MediaMaker raises TypeError for invalid negative_prompt type."""
        mock_model = MockMediaModel()

        with pytest.raises(TypeError, match="`negative_prompt` need be a str or None"):
            MediaMaker(model=mock_model, negative_prompt=123)

    def test_mediamaker_config_invalid_type(self):
        """Test MediaMaker raises TypeError for invalid config type."""
        mock_model = MockMediaModel()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            MediaMaker(model=mock_model, config="invalid")

    def test_mediamaker_forward_with_string(self):
        """Test MediaMaker forward with string prompt."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model)
        result = mediamaker("A beautiful landscape")

        assert result == "generated_media_url"

    def test_mediamaker_forward_with_none_prompt(self):
        """Test MediaMaker raises ValueError for None prompt."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(
            model=mock_model, message_fields={"task_inputs": "nonexistent"}
        )
        msg = Message()

        with pytest.raises(ValueError, match="`prompt` cannot be None"):
            mediamaker(msg)

    def test_mediamaker_process_any_response_type(self):
        """Test MediaMaker processes any response type without raising."""
        mock_model = Mock()
        mock_model.model_type = "text_to_image"
        mock_response = Mock(spec=ModelResponse)
        mock_response.response_type = "image_generation"
        mock_response.consume.return_value = b"image_bytes"

        mediamaker = MediaMaker.__new__(MediaMaker)
        mediamaker._buffers = {
            "model": mock_model,
            "response_mode": None,
            "response_format": None,
        }
        mediamaker._modules = {}
        mediamaker._parameters = {}

        result = mediamaker._process_model_response(mock_response, "test")
        assert result == b"image_bytes"

    @pytest.mark.asyncio
    async def test_mediamaker_aforward_with_string(self):
        """Test MediaMaker async forward with string prompt."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model)
        result = await mediamaker.aforward("A beautiful landscape")

        assert result == "generated_media_url"

    def test_mediamaker_with_hooks(self):
        """Test MediaMaker with hooks."""
        from msgflux.nn.hooks import Guard

        mock_model = MockMediaModel()
        guard = Guard(validator=lambda data: {"safe": True}, on="pre")

        mediamaker = MediaMaker(model=mock_model, hooks=[guard])
        assert len(mediamaker.generator._forward_pre_hooks) == 1

    def test_mediamaker_response_format_url(self):
        """Test MediaMaker with response_format url."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model, response_format="url")

        assert mediamaker.response_format == "url"

    def test_mediamaker_response_format_base64(self):
        """Test MediaMaker with response_format base64."""
        mock_model = MockMediaModel()
        mediamaker = MediaMaker(model=mock_model, response_format="base64")

        assert mediamaker.response_format == "base64"

    def test_mediamaker_guard_short_circuit_returns_message(self):
        """Guard with message= short-circuits forward and returns the message."""
        from msgflux.nn.hooks import Guard

        mock_model = MockMediaModel()
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="pre",
            message="Prompt blocked.",
        )
        mediamaker = MediaMaker(model=mock_model, hooks=[guard])

        result = mediamaker("unsafe prompt")
        assert result == "Prompt blocked."

    def test_mediamaker_guard_raises_without_message(self):
        """Guard without message= raises UnsafeUserInputError."""
        from msgflux.nn.hooks import Guard
        from msgflux.exceptions import UnsafeUserInputError

        mock_model = MockMediaModel()
        guard = Guard(validator=lambda data: {"safe": False}, on="pre")
        mediamaker = MediaMaker(model=mock_model, hooks=[guard])

        with pytest.raises(UnsafeUserInputError):
            mediamaker("unsafe prompt")

    @pytest.mark.asyncio
    async def test_mediamaker_guard_short_circuit_aforward(self):
        """Guard short-circuit also works in aforward."""
        from msgflux.nn.hooks import Guard

        mock_model = MockMediaModel()
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="pre",
            message="Blocked async.",
        )
        mediamaker = MediaMaker(model=mock_model, hooks=[guard])

        result = await mediamaker.aforward("unsafe prompt")
        assert result == "Blocked async."
