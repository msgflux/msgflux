from os import getenv
from typing import Any, Dict, List, Optional, Union

from msgflux.models.httpx import HTTPXModelClient
from msgflux.models.providers.jinaai import JinaAITextReranker
from msgflux.models.providers.openai import (
    OpenAIChatCompletion,
    OpenAISpeechToText,
    OpenAITextEmbedder,
)
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse
from msgflux.models.types import TextClassifierModel
from msgflux.utils.tenacity import model_retry


class _BaseVLLM:
    """Configurations to use vLLM models."""

    provider: str = "vllm"

    def _get_base_url(self):
        base_url = getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        if base_url is None:
            raise ValueError("Please set `VLLM_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("VLLM_API_KEY", "vllm")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class VLLMChatCompletion(_BaseVLLM, OpenAIChatCompletion):
    """vLLM Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        extra_body = params.get("extra_body", {})

        if response_format is not None:
            extra_body["guided_json"] = response_format

        if self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }

        params["extra_body"] = extra_body
        return params

# TODO: moderation based on ChatCompletion
# llama guard prompt models

@register_model
class VLLMTextEmbedder(OpenAITextEmbedder, _BaseVLLM):
    """vLLM Text Embedder."""

@register_model
class VLLMSpeechToText(OpenAISpeechToText, _BaseVLLM):
    """vLLM Speech to Text."""

@register_model
class VLLMTextReranker(JinaAITextReranker, _BaseVLLM):
    """vLLM Text Reranker."""

@register_model
class VLLMTextClassifier(_BaseVLLM, HTTPXModelClient, TextClassifierModel):
    """vLLM Text Score."""

    url_path = "/classify"

    def __init__(self, model_id: str, base_url: Optional[str] = None):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        results = [item["label"] for item in data]
        response.add(results)
        return response

    @model_retry
    def __call__(self, data: Union[str, List[str]]) -> ModelResponse:
        """Args:
        data:
            Input text to classify.
        """
        if isinstance(data, str):
            data = [data]
        response = self._generate(input=data)
        return response
