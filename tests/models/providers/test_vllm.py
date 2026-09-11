"""Tests for msgflux.models.providers.vllm module."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from msgflux.chat_messages import ChatMessages


class TestVLLMProviderImport:
    """Test VLLM provider import and initialization."""

    def test_vllm_import_available(self):
        """Test that VLLM provider imports correctly when dependencies are available."""
        try:
            from msgflux.models.providers.vllm import (
                VLLMChatCompletion,
                VLLMSpeechToText,
                VLLMTextClassifier,
                VLLMTextEmbedder,
                VLLMTextReranker,
            )

            # If we get here, imports worked
            assert True
        except ImportError:
            pytest.skip("VLLM dependencies not available")

    def test_vllm_models_registered(self):
        """Test that VLLM models are registered with @register_model."""

        from msgflux.models.registry import model_registry

        # Check if VLLM models are registered
        if "chat_completion" in model_registry:
            assert "vllm" in model_registry.get("chat_completion", {})

        if "text_embedder" in model_registry:
            assert "vllm" in model_registry.get("text_embedder", {})


class TestVLLMChatCompletion:
    """Test suite for VLLMChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("VLLM_API_KEY", "test-key-12345")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    def test_chat_completion_initialization(self):
        """Test VLLMChatCompletion initialization."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        assert model.model_id == "llama-3"
        assert model.provider == "vllm"
        assert model.model_type == "chat_completion"

    def test_responses_mode_uses_clear_text_reasoning_codec(self):

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="deepseek-r1", api_mode="responses")

        assert model.reasoning_codec.name == "responses_reasoning_text"

    def test_chat_mode_does_not_replay_parsed_reasoning(self):

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="deepseek-r1")
        messages = ChatMessages()
        messages.add_reasoning("private chain")
        messages.add_assistant("answer")

        params = model._build_generation_params(
            messages,
            system_prompt=None,
            prefilling=None,
            tool_catalog=None,
        )

        assert params["messages"] == [{"role": "assistant", "content": "answer"}]

    def test_chat_completion_with_parameters(self):
        """Test VLLMChatCompletion with custom parameters."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(
            model_id="llama-3",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.9,
        )

        assert model.sampling_run_params["max_tokens"] == 1000
        assert model.sampling_run_params["temperature"] == 0.7
        assert model.sampling_run_params["top_p"] == 0.9

    def test_chat_completion_ignores_openai_only_prompt_cache_retention(self):
        """Test VLLMChatCompletion does not forward OpenAI-only cache params."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(
            model_id="llama-3",
            prompt_cache_retention="24h",
        )

        assert "prompt_cache_retention" not in model.sampling_run_params

    def test_chat_completion_ignores_openai_only_logprobs(self):
        """Test VLLMChatCompletion does not forward OpenAI-only logprobs."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(
            model_id="llama-3",
            logprobs=True,
            top_logprobs=2,
        )

        assert "logprobs" not in model.sampling_run_params
        assert "top_logprobs" not in model.sampling_run_params

    def test_chat_completion_base_url(self):
        """Test VLLMChatCompletion uses VLLM_BASE_URL."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        assert "base_url" in model.sampling_params
        assert model.sampling_params["base_url"] == "http://localhost:8000/v1"

    def test_chat_completion_adapt_params_response_format(self):
        """Test VLLMChatCompletion adapts response_format to guided_json."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        params = {"messages": [], "response_format": {"type": "json_object"}}

        adapted = model._adapt_params(params)

        assert "response_format" not in adapted
        assert "extra_body" in adapted
        assert adapted["extra_body"]["guided_json"] == {"type": "json_object"}

    def test_chat_completion_adapt_params_preserves_extra_body(self):
        """Test VLLMChatCompletion copies existing extra_body before extending it."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        extra_body = {"custom": {"value": 1}}
        params = {
            "messages": [],
            "response_format": {"type": "json_object"},
            "extra_body": extra_body,
        }

        adapted = model._adapt_params(params)

        assert adapted["extra_body"]["custom"] == {"value": 1}
        assert adapted["extra_body"]["guided_json"] == {"type": "json_object"}
        assert adapted["extra_body"] is not extra_body
        assert extra_body == {"custom": {"value": 1}}

    def test_chat_completion_adapt_params_enable_thinking(self):
        """Test VLLMChatCompletion adapts enable_thinking parameter."""

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3", enable_thinking=True)

        params = {"messages": []}
        adapted = model._adapt_params(params)

        assert "extra_body" in adapted
        assert "chat_template_kwargs" in adapted["extra_body"]
        assert adapted["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


class TestVLLMTextEmbedder:
    """Test suite for VLLMTextEmbedder."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("VLLM_API_KEY", "test-key-12345")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    def test_text_embedder_initialization(self):
        """Test VLLMTextEmbedder initialization."""

        from msgflux.models.providers.vllm import VLLMTextEmbedder

        model = VLLMTextEmbedder(model_id="bge-m3")

        assert model.model_id == "bge-m3"
        assert model.provider == "vllm"
        assert model.model_type == "text_embedder"

    def test_text_embedder_base_url(self):
        """Test VLLMTextEmbedder uses VLLM_BASE_URL."""

        from msgflux.models.providers.vllm import VLLMTextEmbedder

        model = VLLMTextEmbedder(model_id="bge-m3")

        assert "base_url" in model.sampling_params
        assert model.sampling_params["base_url"] == "http://localhost:8000/v1"


class TestVLLMSpeechToText:
    """Test suite for VLLMSpeechToText."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("VLLM_API_KEY", "test-key-12345")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    def test_speech_to_text_initialization(self):
        """Test VLLMSpeechToText initialization."""
        from msgflux.models.providers.vllm import VLLMSpeechToText

        model = VLLMSpeechToText(model_id="whisper-large")

        assert model.model_id == "whisper-large"
        assert model.provider == "vllm"
        assert model.model_type == "speech_to_text"


class TestVLLMTextClassifier:
    """Test suite for VLLMTextClassifier."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("VLLM_API_KEY", "test-key-12345")
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock HTTPX client."""
        with (
            patch("msgflux.models.httpx.httpx2.Client") as mock_client,
            patch("msgflux.models.httpx.httpx2.AsyncClient") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_text_classifier_initialization(self, mock_httpx_client):
        """Test VLLMTextClassifier initialization."""
        pytest.importorskip("httpx2")

        from msgflux.models.providers.vllm import VLLMTextClassifier

        model = VLLMTextClassifier(model_id="text-classifier")

        assert model.model_id == "text-classifier"
        assert model.provider == "vllm"
        assert model.model_type == "text_classifier"

    def test_text_classifier_base_url(self, mock_httpx_client):
        """Test VLLMTextClassifier uses VLLM_BASE_URL."""
        pytest.importorskip("httpx2")

        from msgflux.models.providers.vllm import VLLMTextClassifier

        model = VLLMTextClassifier(model_id="text-classifier")

        assert "base_url" in model.sampling_params
        assert model.sampling_params["base_url"] == "http://localhost:8000/v1"

    def test_text_classifier_custom_base_url(self, mock_httpx_client):
        """Test VLLMTextClassifier with custom base_url."""
        pytest.importorskip("httpx2")

        from msgflux.models.providers.vllm import VLLMTextClassifier

        custom_url = "http://custom-vllm.example.com/v1"
        model = VLLMTextClassifier(model_id="text-classifier", base_url=custom_url)

        assert model.sampling_params["base_url"] == custom_url

    def test_text_classifier_endpoint(self, mock_httpx_client):
        """Test VLLMTextClassifier has correct endpoint."""
        pytest.importorskip("httpx2")

        from msgflux.models.providers.vllm import VLLMTextClassifier

        model = VLLMTextClassifier(model_id="text-classifier")

        assert model.endpoint == "/classify"


class TestVLLMBaseURL:
    """Test suite for VLLM base_url handling."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("VLLM_API_KEY", "test-key-12345")

    def test_default_base_url(self, monkeypatch):
        """Test default VLLM base URL when not set."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        # Should default to localhost:8000/v1
        assert model.sampling_params["base_url"] == "http://localhost:8000/v1"

    def test_custom_base_url(self, monkeypatch):
        """Test custom VLLM base URL from environment."""
        custom_url = "http://vllm.example.com:8080/v1"
        monkeypatch.setenv("VLLM_BASE_URL", custom_url)

        from msgflux.models.providers.vllm import VLLMChatCompletion

        model = VLLMChatCompletion(model_id="llama-3")

        assert model.sampling_params["base_url"] == custom_url
