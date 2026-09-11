"""Tests for direct multipart OpenAI image edits."""

from __future__ import annotations

import httpx2
import pytest

from msgflux.models.http_transport import HTTPTransport


def _image_edit_payload() -> dict:
    return {
        "created": 1_780_000_002,
        "background": "opaque",
        "data": [{"b64_json": "edited-image"}],
        "output_format": "png",
        "quality": "low",
        "size": "1024x1024",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"text_tokens": 4, "image_tokens": 6},
            "output_tokens": 20,
            "output_tokens_details": {"text_tokens": 0, "image_tokens": 20},
            "total_tokens": 30,
        },
    }


def test_openai_image_edit_uses_replayable_multipart(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_image_edit_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAIImageTextToImage

    model = OpenAIImageTextToImage(
        model_id="gpt-image-2",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(
        "Add a blue circle",
        [b"first-image", b"second-image"],
        mask=b"mask-image",
        response_format="base64",
        n=2,
        background="opaque",
        input_fidelity="high",
        output_compression=80,
        output_format="webp",
        quality="low",
        size="1024x1024",
    )

    request = requests[0]
    body = request.content
    assert request.url == "https://api.openai.com/v1/images/edits"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["content-type"].startswith("multipart/form-data;")
    assert body.count(b'name="image[]"') == 2
    assert b'name="mask"' in body
    assert b'name="prompt"' in body and b"Add a blue circle" in body
    assert b'name="model"' in body and b"gpt-image-2" in body
    assert b'name="n"' in body and b"2" in body
    assert b'name="response_format"' in body and b"b64_json" in body
    assert b'name="background"' in body and b"opaque" in body
    assert b'name="input_fidelity"' in body and b"high" in body
    assert b'name="output_compression"' in body and b"80" in body
    assert b'name="output_format"' in body and b"webp" in body
    assert b'name="quality"' in body and b"low" in body
    assert b'name="size"' in body and b"1024x1024" in body
    assert response.consume() == "edited-image"
    assert response.metadata.usage.input_tokens_details.image_tokens == 6
    assert response.metadata.details.created == 1_780_000_002
    client.close()


@pytest.mark.asyncio
async def test_openai_image_edit_async_uses_async_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_image_edit_payload())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAIImageTextToImage

    model = OpenAIImageTextToImage(
        model_id="gpt-image-2",
        http_transport=HTTPTransport(async_client=client),
        retry=False,
    )
    response = await model.acall("Add a blue circle", b"source-image")

    assert response.consume() == "edited-image"
    assert requests[0].content.count(b'name="image[]"') == 1
    await client.aclose()


def test_openai_image_edit_resolves_credentials_on_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, json=_image_edit_payload())
        )
    )

    from msgflux.models.providers.openai import OpenAIImageTextToImage

    model = OpenAIImageTextToImage(
        model_id="gpt-image-2",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model("Edit this", b"source-image")
    client.close()


def test_openai_image_edit_does_not_serialize_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from msgflux.models.providers.openai import OpenAIImageTextToImage

    model = OpenAIImageTextToImage(
        model_id="gpt-image-2",
        http_transport=HTTPTransport(),
        retry=False,
    )

    assert "http_transport" not in model.serialize()["state"]


def test_openai_image_edit_retries_with_replayable_files(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bodies = []

    def handler(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx2.Response(429, headers={"retry-after": "0"}, json={})
        return httpx2.Response(200, json=_image_edit_payload())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAIImageTextToImage

    model = OpenAIImageTextToImage(
        model_id="gpt-image-2",
        http_transport=HTTPTransport(client=client, max_retries=1),
        retry=False,
    )
    response = model("Edit this", b"source-image")

    assert response.consume() == "edited-image"
    assert len(bodies) == 2
    assert all(b"source-image" in body for body in bodies)
    client.close()
