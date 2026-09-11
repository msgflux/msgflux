"""Tests for direct OpenAI-compatible image generation."""

from __future__ import annotations

import httpx2
import msgspec
import pytest

from msgflux.models.http_transport import HTTPTransport


def _image_payload(*, data=None, usage=True) -> dict:
    payload = {
        "created": 1_780_000_000,
        "background": "opaque",
        "data": data or [{"b64_json": "encoded-image"}],
        "output_format": "png",
        "quality": "low",
        "size": "1024x1024",
    }
    if usage:
        payload["usage"] = {
            "input_tokens": 12,
            "input_tokens_details": {"text_tokens": 4, "image_tokens": 8},
            "output_tokens": 20,
            "output_tokens_details": {"text_tokens": 0, "image_tokens": 20},
            "total_tokens": 32,
        }
    return payload


def test_openai_image_generation_uses_direct_http_and_normalizes_metadata(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_image_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAITextToImage

    model = OpenAITextToImage(
        model_id="gpt-image-1-mini",
        moderation="low",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(
        "A blue circle",
        response_format="base64",
        size="1024x1024",
        quality="low",
        background="opaque",
    )

    assert response.response_type == "image_generation"
    assert response.consume() == "encoded-image"
    assert response.metadata.usage.input_tokens == 12
    assert response.metadata.usage.input_tokens_details.text_tokens == 4
    assert response.metadata.usage.input_tokens_details.image_tokens == 8
    assert response.metadata.usage.output_tokens_details.image_tokens == 20
    assert response.metadata.details == {
        "created": 1_780_000_000,
        "size": "1024x1024",
        "quality": "low",
        "output_format": "png",
        "background": "opaque",
    }
    assert requests[0].url == "https://api.openai.com/v1/images/generations"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert msgspec.json.decode(requests[0].content) == {
        "background": "opaque",
        "model": "gpt-image-1-mini",
        "moderation": "low",
        "n": 1,
        "prompt": "A blue circle",
        "quality": "low",
        "response_format": "b64_json",
        "size": "1024x1024",
    }
    client.close()


@pytest.mark.asyncio
async def test_openai_image_generation_async_preserves_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json=_image_payload(
                data=[{"url": "https://example.com/generated.png"}],
                usage=False,
            ),
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAITextToImage

    model = OpenAITextToImage(
        model_id="dall-e-3",
        http_transport=HTTPTransport(async_client=client),
        retry=False,
    )
    response = await model.acall("A blue circle", response_format="url")

    assert response.consume() == "https://example.com/generated.png"
    assert response.metadata.usage is None
    assert msgspec.json.decode(requests[0].content) == {
        "model": "dall-e-3",
        "n": 1,
        "prompt": "A blue circle",
        "response_format": "url",
    }
    await client.aclose()


def test_openai_image_generation_preserves_multiple_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200,
                json=_image_payload(
                    data=[
                        {"b64_json": "first"},
                        {
                            "url": "https://example.com/second.png",
                            "b64_json": "second",
                        },
                    ]
                ),
            )
        )
    )

    from msgflux.models.providers.openai import OpenAITextToImage

    model = OpenAITextToImage(
        model_id="gpt-image-1-mini",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    assert model("Three values", n=2).consume() == [
        "first",
        "https://example.com/second.png",
        "second",
    ]
    client.close()


def test_image_router_uses_its_own_url_and_credentials(monkeypatch):
    monkeypatch.setenv("IMAGEROUTER_API_KEY", "router-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_image_payload(usage=False))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    import msgflux as mf

    model = mf.Model.text_to_image(
        "imagerouter/default",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    model("A blue circle")

    assert requests[0].url == (
        "https://api.imagerouter.io/v1/openai/images/generations"
    )
    assert requests[0].headers["authorization"] == "Bearer router-key"
    client.close()


def test_openai_image_generation_resolves_credentials_on_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=_image_payload())
        )
    )

    from msgflux.models.providers.openai import OpenAITextToImage

    model = OpenAITextToImage(
        model_id="gpt-image-1-mini",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model("A blue circle")
    client.close()


def test_openai_image_generation_does_not_serialize_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from msgflux.models.providers.openai import OpenAITextToImage

    model = OpenAITextToImage(
        model_id="gpt-image-1-mini",
        http_transport=HTTPTransport(),
        retry=False,
    )

    assert "http_transport" not in model.serialize()["state"]
