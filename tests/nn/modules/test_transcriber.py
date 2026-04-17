"""Tests for msgflux.nn.modules.transcriber module."""

import pytest
from unittest.mock import Mock, patch

from msgflux.nn.modules.transcriber import Transcriber
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockSTTModel(BaseModel):
    """Mock speech-to-text model for testing."""

    model_type = "speech_to_text"
    provider = "mock"

    def __init__(self):
        self.model_id = "test-stt"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("transcript")
        response.add("Hello world")
        return response

    async def acall(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("transcript")
        response.add("Hello world")
        return response


class TestTranscriber:
    """Test suite for Transcriber module."""

    def test_transcriber_initialization(self):
        """Test Transcriber basic initialization."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)

        assert transcriber.model is mock_model
        assert transcriber.response_format == "text"

    def test_transcriber_string_model_calls_speech_to_text(self):
        """Passing a string model must call Model.speech_to_text."""
        mock_model = MockSTTModel()

        with patch(
            "msgflux.nn.modules.transcriber.Model.speech_to_text",
            return_value=mock_model,
        ) as mock_factory:
            transcriber = Transcriber(model="openai/whisper-1")

        mock_factory.assert_called_once_with("openai/whisper-1")
        assert transcriber.model is mock_model

    def test_transcriber_initialization_with_config(self):
        """Test Transcriber initialization with configuration."""
        mock_model = MockSTTModel()
        config = {"language": "en", "stream": False}
        transcriber = Transcriber(model=mock_model, config=config)

        assert transcriber._buffers["config"]["language"] == "en"
        assert transcriber._buffers["config"]["stream"] is False

    def test_transcriber_invalid_model_type(self):
        """Test Transcriber raises TypeError for invalid model type."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        with pytest.raises(TypeError, match="need be a `speech_to_text` model"):
            Transcriber(model=mock_model)

    def test_transcriber_string_model_setter_calls_speech_to_text(self):
        """Assigning a string to transcriber.model must call Model.speech_to_text."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)

        new_mock = MockSTTModel()

        with patch(
            "msgflux.nn.modules.transcriber.Model.speech_to_text",
            return_value=new_mock,
        ) as mock_factory:
            transcriber.model = "openai/whisper-1"

        mock_factory.assert_called_once_with("openai/whisper-1")
        assert transcriber.model is new_mock

    def test_transcriber_with_response_format(self):
        """Test Transcriber with custom response_format."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model, response_format="json")

        assert transcriber.response_format == "json"

    def test_transcriber_invalid_response_format(self):
        """Test Transcriber raises ValueError for invalid response_format."""
        mock_model = MockSTTModel()

        with pytest.raises(ValueError, match="`response_format` can be"):
            Transcriber(model=mock_model, response_format="invalid")

    def test_transcriber_response_format_invalid_type(self):
        """Test Transcriber raises TypeError for non-string response_format."""
        mock_model = MockSTTModel()

        with pytest.raises(TypeError, match="`response_format` need be a str"):
            Transcriber(model=mock_model, response_format=123)

    def test_transcriber_with_prompt(self):
        """Test Transcriber initialization with prompt."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model, prompt="Transcribe clearly")

        assert transcriber.prompt == "Transcribe clearly"

    def test_transcriber_config_invalid_type(self):
        """Test Transcriber raises TypeError for invalid config type."""
        mock_model = MockSTTModel()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            Transcriber(model=mock_model, config="invalid")

    def test_transcriber_forward_with_bytes(self):
        """Test Transcriber forward with bytes input."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)
        result = transcriber(b"audio_data")

        assert result == "Hello world"

    def test_transcriber_forward_with_string_path(self):
        """Test Transcriber forward with string path input."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)
        result = transcriber("audio.mp3")

        assert result == "Hello world"

    def test_transcriber_forward_with_dict_input(self):
        """Test Transcriber forward with dict input."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)
        result = transcriber({"audio": "audio.mp3"})

        assert result == "Hello world"

    def test_transcriber_invalid_response_type(self):
        """Test Transcriber raises ValueError for unsupported response type."""
        mock_model = Mock()
        mock_model.model_type = "speech_to_text"
        mock_response = Mock(spec=ModelResponse)
        mock_response.response_type = "text_generation"
        mock_model.return_value = mock_response

        transcriber = Transcriber(model=mock_model)
        with pytest.raises(ValueError, match="Unsupported model response type"):
            transcriber(b"audio")

    @pytest.mark.asyncio
    async def test_transcriber_aforward_with_bytes(self):
        """Test Transcriber async forward with bytes input."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model)
        result = await transcriber.aforward(b"audio_data")

        assert result == "Hello world"

    def test_transcriber_supported_formats(self):
        """Test all supported response formats."""
        formats = ["json", "text", "srt", "verbose_json", "vtt"]
        for fmt in formats:
            mock_model = MockSTTModel()
            transcriber = Transcriber(model=mock_model, response_format=fmt)
            assert transcriber.response_format == fmt

    def test_transcriber_with_response_template(self):
        """Test Transcriber with response_template."""
        mock_model = MockSTTModel()
        transcriber = Transcriber(model=mock_model, response_template="{{content}}")
        assert transcriber.response_template == "{{content}}"
