"""Tests for the direct OpenAI Moderations transport."""

from __future__ import annotations

import httpx2
import msgspec
import pytest

from msgflux.models.http_transport import HTTPTransport


def _moderation_payload(*, flagged: bool = False) -> dict:
    return {
        "id": "modr_test",
        "model": "omni-moderation-latest",
        "results": [
            {
                "flagged": flagged,
                "categories": {
                    "harassment": flagged,
                    "violence": False,
                },
                "category_scores": {
                    "harassment": 0.9 if flagged else 0.01,
                    "violence": 0.02,
                },
                "category_applied_input_types": {
                    "harassment": ["text"],
                    "violence": ["text", "image"],
                },
            }
        ],
    }


def test_openai_moderation_uses_direct_http_and_preserves_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_moderation_payload(flagged=True))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(client=client)

    from msgflux.models.providers.openai import OpenAIModeration

    model = OpenAIModeration(
        model_id="omni-moderation-latest",
        http_transport=transport,
        retry=False,
    )
    response = model("test input")

    result = response.consume()
    assert response.response_type == "moderation"
    assert result.safe is False
    assert result.results.flagged is True
    assert result.results.categories.harassment is True
    assert result.results.category_scores.harassment == 0.9
    assert result.results.category_applied_input_types.violence == ["text", "image"]
    assert requests[0].url == "https://api.openai.com/v1/moderations"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert msgspec.json.decode(requests[0].content) == {
        "input": "test input",
        "model": "omni-moderation-latest",
    }
    client.close()


@pytest.mark.asyncio
async def test_openai_moderation_async_multimodal_request_is_cached(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_moderation_payload())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(async_client=client)

    from msgflux.models.providers.openai import OpenAIModeration

    model = OpenAIModeration(
        model_id="omni-moderation-latest",
        enable_cache=True,
        http_transport=transport,
        retry=False,
    )
    inputs = [
        {"type": "text", "text": "Check this image."},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"},
        },
    ]
    first = await model.acall(inputs)
    second = await model.acall(inputs)

    assert first.consume().safe is True
    assert second is first
    assert len(requests) == 1
    assert msgspec.json.decode(requests[0].content) == {
        "input": inputs,
        "model": "omni-moderation-latest",
    }
    await client.aclose()


def test_openai_moderation_resolves_credentials_on_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=_moderation_payload())
        )
    )

    from msgflux.models.providers.openai import OpenAIModeration

    model = OpenAIModeration(
        model_id="omni-moderation-latest",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model("test input")
    client.close()


def test_openai_moderation_rejects_response_without_results(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200,
                json={"id": "modr_test", "model": "omni-moderation-latest"},
            )
        )
    )

    from msgflux.models.providers.openai import OpenAIModeration

    model = OpenAIModeration(
        model_id="omni-moderation-latest",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="did not contain results"):
        model("test input")
    client.close()


def test_openai_moderation_does_not_serialize_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from msgflux.models.providers.openai import OpenAIModeration

    model = OpenAIModeration(
        model_id="omni-moderation-latest",
        http_transport=HTTPTransport(),
        retry=False,
    )

    assert "http_transport" not in model.serialize()["state"]
