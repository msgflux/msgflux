"""Tests for msgflux.models.providers.openai module."""

import os
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import msgspec
import pytest


class TestOpenAIProviderImport:
    """Test OpenAI provider import and initialization."""

    def test_openai_import_available(self):
        """Test that OpenAI provider imports correctly when dependencies are available."""
        try:
            from msgflux.models.providers.openai import (
                OpenAIChatCompletion,
                OpenAIModeration,
                OpenAISpeechToText,
                OpenAITextEmbedder,
                OpenAITextToImage,
                OpenAITextToSpeech,
            )

            # If we get here, imports worked
            assert True
        except ImportError:
            pytest.skip("OpenAI dependencies not available")

    def test_openai_models_registered(self):
        """Test that OpenAI models are registered with @register_model."""
        pytest.importorskip("openai", reason="openai not installed")

        from msgflux.models.registry import model_registry

        # Check if OpenAI models are registered
        if "chat_completion" in model_registry:
            assert "openai" in model_registry.get("chat_completion", {})

        if "text_to_speech" in model_registry:
            assert "openai" in model_registry.get("text_to_speech", {})


class TestOpenAIChatCompletion:
    """Test suite for OpenAIChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_initialization(self, mock_openai_client):
        """Test OpenAIChatCompletion initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4")

        assert model.model_id == "gpt-4"
        assert model.provider == "openai"
        assert model.model_type == "chat_completion"

    def test_chat_completion_with_parameters(self, mock_openai_client):
        """Test OpenAIChatCompletion with custom parameters."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="gpt-4",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )

        assert model.sampling_run_params["max_tokens"] == 1000
        assert model.sampling_run_params["temperature"] == 0.7
        assert model.sampling_run_params["top_p"] == 0.9

    def test_chat_completion_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        pytest.importorskip("openai")

        # Remove API key
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from msgflux.models.providers.openai import OpenAIChatCompletion

        with pytest.raises(ValueError, match="OpenAI key is not available"):
            OpenAIChatCompletion(model_id="gpt-4")

    def test_chat_completion_with_reasoning_params(self, mock_openai_client):
        """Test OpenAIChatCompletion with reasoning parameters."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(
            model_id="o1-preview",
            reasoning_effort="high",
            enable_thinking=True,
            return_reasoning=True,
        )

        assert model.sampling_run_params.get("reasoning_effort") == "high"
        assert model.enable_thinking is True
        assert model.return_reasoning is True

    def test_chat_completion_adapt_params(self, mock_openai_client):
        """Test parameter adaptation for OpenAI."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        model = OpenAIChatCompletion(model_id="gpt-4", max_tokens=100)

        # Test adaptation
        params = {"max_tokens": 100, "messages": []}
        adapted = model._adapt_params(params)

        assert "max_completion_tokens" in adapted
        assert "max_tokens" not in adapted

    def test_prepare_generate_kwargs_lowers_dict_schema(self, mock_openai_client):
        """Test OpenAI transport schema lowering for dict-based structured outputs."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        kwargs = {"typed_parser": None, "generation_schema": DictOutput}

        (
            typed_parser,
            generation_schema,
            transport_generation_schema,
        ) = model._prepare_generate_kwargs(kwargs)

        assert typed_parser is None
        assert generation_schema is DictOutput
        assert transport_generation_schema is not DictOutput
        assert kwargs["response_format"]["json_schema"]["schema"]["properties"][
            "entities"
        ]["items"]["properties"]["entries"]["type"] == "array"

    def test_process_completion_model_output_restores_dict_shape(
        self, mock_openai_client
    ):
        """Test transport-schema decoding is restored to the logical dict shape."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        class DictOutput(msgspec.Struct):
            entities: List[Dict[str, str]]

        model = OpenAIChatCompletion(model_id="gpt-4")
        transport_generation_schema = model._prepare_generate_kwargs(
            {"typed_parser": None, "generation_schema": DictOutput}
        )[2]

        model_output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"entities":[{"entries":[{"key":"name","value":"Apple"},{"key":"type","value":"Organization"}]}]}',
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(
            model_output,
            generation_schema=DictOutput,
            transport_generation_schema=transport_generation_schema,
        )

        assert response.response_type == "structured"
        assert response.data == {
            "entities": [{"name": "Apple", "type": "Organization"}]
        }


class TestOpenAITextToSpeech:
    """Test suite for OpenAITextToSpeech."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_to_speech_initialization(self, mock_openai_client):
        """Test OpenAITextToSpeech initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(model_id="tts-1")

        assert model.model_id == "tts-1"
        assert model.provider == "openai"
        assert model.model_type == "text_to_speech"

    def test_text_to_speech_with_voice_and_speed(self, mock_openai_client):
        """Test OpenAITextToSpeech with voice and speed parameters."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToSpeech

        model = OpenAITextToSpeech(
            model_id="tts-1",
            voice="nova",
            speed=1.5,
        )

        assert model.sampling_run_params["voice"] == "nova"
        assert model.sampling_run_params["speed"] == 1.5


class TestOpenAITextToImage:
    """Test suite for OpenAITextToImage."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_to_image_initialization(self, mock_openai_client):
        """Test OpenAITextToImage initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToImage

        model = OpenAITextToImage(model_id="dall-e-3")

        assert model.model_id == "dall-e-3"
        assert model.provider == "openai"
        assert model.model_type == "text_to_image"

    def test_text_to_image_with_moderation(self, mock_openai_client):
        """Test OpenAITextToImage with moderation parameter."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextToImage

        model = OpenAITextToImage(
            model_id="dall-e-3",
            moderation="low",
        )

        assert model.sampling_run_params.get("moderation") == "low"


class TestOpenAISpeechToText:
    """Test suite for OpenAISpeechToText."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_speech_to_text_initialization(self, mock_openai_client):
        """Test OpenAISpeechToText initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(model_id="whisper-1")

        assert model.model_id == "whisper-1"
        assert model.provider == "openai"
        assert model.model_type == "speech_to_text"

    def test_speech_to_text_with_temperature(self, mock_openai_client):
        """Test OpenAISpeechToText with temperature parameter."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAISpeechToText

        model = OpenAISpeechToText(
            model_id="whisper-1",
            temperature=0.5,
        )

        assert model.sampling_run_params["temperature"] == 0.5


class TestOpenAITextEmbedder:
    """Test suite for OpenAITextEmbedder."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_embedder_initialization(self, mock_openai_client):
        """Test OpenAITextEmbedder initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextEmbedder

        model = OpenAITextEmbedder(model_id="text-embedding-3-large")

        assert model.model_id == "text-embedding-3-large"
        assert model.provider == "openai"
        assert model.model_type == "text_embedder"

    def test_text_embedder_with_dimensions(self, mock_openai_client):
        """Test OpenAITextEmbedder with dimensions parameter."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextEmbedder

        model = OpenAITextEmbedder(
            model_id="text-embedding-3-large",
            dimensions=1536,
        )

        assert model.sampling_run_params["dimensions"] == 1536


class TestOpenAIModeration:
    """Test suite for OpenAIModeration."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_moderation_initialization(self, mock_openai_client):
        """Test OpenAIModeration initialization."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIModeration

        model = OpenAIModeration(model_id="omni-moderation-latest")

        assert model.model_id == "omni-moderation-latest"
        assert model.provider == "openai"
        assert model.model_type == "moderation"


class TestOpenAIBaseURL:
    """Test suite for custom base_url parameter."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_custom_base_url(self, mock_openai_client):
        """Test OpenAIChatCompletion with custom base_url."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAIChatCompletion

        custom_url = "https://custom-api.example.com"
        model = OpenAIChatCompletion(
            model_id="gpt-4",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url

    def test_text_embedder_custom_base_url(self, mock_openai_client):
        """Test OpenAITextEmbedder with custom base_url."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openai import OpenAITextEmbedder

        custom_url = "https://custom-api.example.com"
        model = OpenAITextEmbedder(
            model_id="text-embedding-ada-002",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url
