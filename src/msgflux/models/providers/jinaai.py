from os import getenv
from typing import List, Optional, Union

from msgflux.models.httpx import HTTPXModelClient
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse
from msgflux.models.types import (
    ImageClassifierModel,
    ImageEmbedderModel,
    TextClassifierModel,
    TextEmbedderModel,
    TextRerankerModel,
)
from msgflux.utils.tenacity import model_retry


class _BaseJinaAI:
    """Configurations to use JinaAI models."""

    provider: str = "jinaai"

    def _get_base_url(self):
        base_url = getenv("JINAAI_BASE_URL", "https://api.jina.ai/v1")
        if base_url is None:
            raise ValueError("Please set `JINAAI_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("JINAAI_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class JinaAITextReranker(_BaseJinaAI, HTTPXModelClient, TextRerankerModel):
    """JinaAI Text Reranker."""

    url_path = "/rerank"

    def __init__(self, model_id: str, base_url: Optional[str] = None):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_reranked")
        model_output = self._execute(**kwargs)
        response.add(model_output["results"])
        return response

    @model_retry
    def __call__(self, query: str, documents: List[str]) -> ModelResponse:
        """Args:
        query:
            Reference text to search for similar.
        documents:
            A list of documents to be ranked.
        """
        response = self._generate(query=query, documents=documents)
        return response

@register_model
class JinaAITextEmbedder(TextEmbedderModel, HTTPXModelClient, _BaseJinaAI):
    """JinaAI Text Embedder."""

    url_path: str = "/embeddings"
    batch_support = True

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
        base_url:
            URL to model provider.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self._initialize()
        self._get_api_key()

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        embedding = [item["embedding"] for item in data]
        if len(embedding) == 1:  # Compatibility
            embedding = embedding[0]
        response.add(embedding)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        """Args:
        data:
            Input text to embed.
        """
        if isinstance(data, str):
            data = [data]
        inputs = [{"text": item} for item in data]
        response = self._generate(input=inputs)
        return response

@register_model
class JinaAIImageEmbedder(ImageEmbedderModel, JinaAITextEmbedder):
    """JinaAI Image Embedder."""

    url_path: str = "/embeddings"
    batch_support = True

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_embedding")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        embedding = [item["embedding"] for item in data]
        if len(embedding) == 1:  # Compatibility
            embedding = embedding[0]
        response.add(embedding)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        """Args:
        data:
            Input image to embed.
        """
        if isinstance(data, str):
            data = [data]
        inputs = [{"image": item} for item in data]
        response = self._generate(input=inputs)
        return response

@register_model
class JinaAITextClassifier(TextClassifierModel, HTTPXModelClient, _BaseJinaAI):
    """JinaAI Text Classifier."""

    url_path: str = "/classify"
    batch_support = True

    def __init__(self, model_id, labels: List[str], base_url: Optional[str] = None):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"labels": labels}
        self._initialize()
        self._get_api_key()

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        pred = [{"label": item["prediction"], "score": item["score"]} for item in data]
        if len(pred) == 1:  # Compatibility
            pred = pred[0]
        response.add(pred)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        """Args:
        data:
            Input text to classify.
        """
        if isinstance(data, str):
            data = [data]
        inputs = [{"text": item} for item in data]
        response = self._generate(input=inputs)
        return response

@register_model
class JinaAIImageClassifier(JinaAITextClassifier, ImageClassifierModel):
    """JinaAI Image Classifier."""

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_classification")
        model_output = self._execute(**kwargs)
        data = model_output["data"]
        pred = [{"label": item["prediction"], "score": item["score"]} for item in data]
        if len(pred) == 1:  # Compatibility
            pred = pred[0]
        response.add(pred)
        return response

    @model_retry
    def __call__(
        self,
        data: Union[str, List[str]],
    ) -> ModelResponse:
        """Args:
        data:
            Input image to embed.
        """
        if isinstance(data, str):
            data = [data]
        inputs = [{"image": item} for item in data]
        response = self._generate(input=inputs)
        return response
