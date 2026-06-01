"""Tests for msgflux.models.providers.kokoro module."""

from unittest.mock import patch

import pytest


class TestKokoroTextToSpeech:
    """Test suite for Kokoro OpenAI-compatible TTS provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client used by the OpenAI-compatible provider."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_kokoro_text_to_speech_registered(self):
        """Kokoro should register as a text_to_speech provider."""
        pytest.importorskip("openai")

        from msgflux.models.registry import model_registry

        assert "kokoro" in model_registry.get("text_to_speech", {})

    def test_kokoro_text_to_speech_initialization(
        self, mock_openai_client, monkeypatch
    ):
        """Kokoro TTS should use local OpenAI-compatible defaults."""
        pytest.importorskip("openai")
        monkeypatch.delenv("KOKORO_BASE_URL", raising=False)
        monkeypatch.delenv("KOKORO_API_KEY", raising=False)

        from msgflux.models.providers.kokoro import KokoroTextToSpeech

        model = KokoroTextToSpeech(model_id="kokoro")

        assert model.model_id == "kokoro"
        assert model.provider == "kokoro"
        assert model.model_type == "text_to_speech"
        assert model.sampling_params["base_url"] == "http://localhost:8880/v1"
        assert model.sampling_run_params["voice"] == "af_heart"
        assert model.sampling_run_params["speed"] == 1.0
        assert model.stream_chunk_size == 1024

        mock_client, mock_async_client = mock_openai_client
        assert mock_client.call_args.kwargs["api_key"] == "not-needed"
        assert mock_async_client.call_args.kwargs["api_key"] == "not-needed"

    def test_kokoro_text_to_speech_uses_env_configuration(
        self, mock_openai_client, monkeypatch
    ):
        """Kokoro TTS should support env-configured URL and API key."""
        pytest.importorskip("openai")
        monkeypatch.setenv("KOKORO_BASE_URL", "http://kokoro.example/v1")
        monkeypatch.setenv("KOKORO_API_KEY", "local-key")

        from msgflux.models.providers.kokoro import KokoroTextToSpeech

        model = KokoroTextToSpeech(
            model_id="kokoro",
            voice="pm_alex",
            stream_chunk_size=512,
        )

        assert model.sampling_params["base_url"] == "http://kokoro.example/v1"
        assert model.sampling_run_params["voice"] == "pm_alex"
        assert model.stream_chunk_size == 512

        mock_client, mock_async_client = mock_openai_client
        assert mock_client.call_args.kwargs["api_key"] == "local-key"
        assert mock_async_client.call_args.kwargs["api_key"] == "local-key"

    def test_kokoro_text_to_speech_base_url_argument_overrides_env(
        self, mock_openai_client, monkeypatch
    ):
        """Explicit base_url should override KOKORO_BASE_URL."""
        pytest.importorskip("openai")
        monkeypatch.setenv("KOKORO_BASE_URL", "http://env.example/v1")

        from msgflux.models.providers.kokoro import KokoroTextToSpeech

        model = KokoroTextToSpeech(
            model_id="kokoro",
            base_url="http://argument.example/v1",
        )

        assert model.sampling_params["base_url"] == "http://argument.example/v1"

    def test_kokoro_text_to_speech_factory(self, mock_openai_client, monkeypatch):
        """Model.text_to_speech should instantiate Kokoro by provider path."""
        pytest.importorskip("openai")
        monkeypatch.delenv("KOKORO_BASE_URL", raising=False)
        monkeypatch.delenv("KOKORO_API_KEY", raising=False)

        import msgflux as mf

        model = mf.Model.text_to_speech(
            "kokoro/kokoro",
            voice="af_bella",
            stream_chunk_size=256,
        )

        assert model.provider == "kokoro"
        assert model.model_id == "kokoro"
        assert model.sampling_run_params["voice"] == "af_bella"
        assert model.stream_chunk_size == 256
