"""Tests for the direct OpenAI-compatible embeddings transport."""

from __future__ import annotations

import httpx2
import msgspec
import pytest

from msgflux.models.http_transport import HTTPTransport


def _embedding_payload(*, dimensions: int = 3) -> dict:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1] * dimensions,
                "index": 0,
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }


def test_openai_embedder_uses_direct_http_and_normalizes_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_embedding_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(client=client)

    from msgflux.models.providers.openai import OpenAITextEmbedder

    model = OpenAITextEmbedder(
        model_id="text-embedding-3-small",
        http_transport=transport,
        retry=False,
    )
    response = model("hello")

    assert response.consume() == [[0.1, 0.1, 0.1]]
    assert response.metadata.usage.input_tokens == 2
    assert response.metadata.usage.output_tokens == 0
    assert response.metadata.usage.total_tokens == 2
    assert response.metadata.usage.raw.prompt_tokens == 2
    assert requests[0].url == "https://api.openai.com/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert msgspec.json.decode(requests[0].content) == {
        "input": "hello",
        "model": "text-embedding-3-small",
    }
    client.close()


@pytest.mark.asyncio
async def test_openai_embedder_async_includes_dimensions_and_caches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_embedding_payload(dimensions=2))

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(async_client=client)

    from msgflux.models.providers.openai import OpenAITextEmbedder

    model = OpenAITextEmbedder(
        model_id="text-embedding-3-small",
        dimensions=2,
        enable_cache=True,
        http_transport=transport,
        retry=False,
    )
    first = await model.acall(["hello"])
    second = await model.acall(["hello"])

    assert first.consume() == [[0.1, 0.1]]
    assert second is first
    assert len(requests) == 1
    assert msgspec.json.decode(requests[0].content) == {
        "dimensions": 2,
        "input": ["hello"],
        "model": "text-embedding-3-small",
    }
    await client.aclose()


def test_openai_embedder_resolves_credentials_on_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from msgflux.models.providers.openai import OpenAITextEmbedder

    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=_embedding_payload())
        )
    )
    model = OpenAITextEmbedder(
        model_id="text-embedding-3-small",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model("hello")
    client.close()


@pytest.mark.parametrize(
    ("model_path", "env", "expected_url", "expected_token"),
    [
        (
            "together/test-embedder",
            {"TOGETHER_API_KEY": "together-key"},
            "https://api.together.xyz/v1/embeddings",
            "together-key",
        ),
        (
            "ollama/test-embedder",
            {"OLLAMA_API_KEY": "ollama-key"},
            "http://localhost:11434/v1/embeddings",
            "ollama-key",
        ),
        (
            "vllm/test-embedder",
            {"VLLM_API_KEY": "vllm-key"},
            "http://localhost:8000/v1/embeddings",
            "vllm-key",
        ),
    ],
)
def test_compatible_embedders_use_provider_url_and_credentials(
    monkeypatch,
    model_path,
    env,
    expected_url,
    expected_token,
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_embedding_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(client=client)

    import msgflux as mf

    model = mf.Model.text_embedder(
        model_path,
        http_transport=transport,
        retry=False,
    )
    model("hello")

    assert str(requests[0].url) == expected_url
    assert requests[0].headers["authorization"] == f"Bearer {expected_token}"
    client.close()


def test_openai_embedder_does_not_serialize_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from msgflux.models.providers.openai import OpenAITextEmbedder

    model = OpenAITextEmbedder(
        model_id="text-embedding-3-small",
        http_transport=HTTPTransport(),
        retry=False,
    )

    serialized = model.serialize()

    assert "http_transport" not in serialized["state"]
