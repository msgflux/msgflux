import pytest

from msgflux.channels import ChannelRegistry
from msgflux.channels.http.app import create_app


class FakeAgent:
    name = "support"

    async def acall(self, **kwargs):
        assert kwargs["stream"] is False
        assert kwargs["vars"] == {"tenant": "acme"}
        return "msgflux server ok"


def test_chat_completions_route_with_msgspec_response():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hello"}],
            "run_config": {"vars": {"tenant": "acme"}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "msgflux server ok"
