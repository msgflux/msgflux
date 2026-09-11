"""Test transports for provider-independent chat model tests."""

from msgflux.models.chat_api import ChatTransport


class EndpointMockTransport(ChatTransport):
    """Route prepared requests to configurable endpoint-shaped mocks."""

    def __init__(self, client, async_client):
        self.client = client
        self.async_client = async_client

    def create(self, owner, request):
        if request.endpoint == "/responses":
            return self.client.responses.create(**request.params)
        if request.endpoint == "/responses/input_tokens":
            return self.client.responses.input_tokens.count(**request.params)
        if request.endpoint == "/responses/compact":
            return self.client.responses.compact(**request.params)
        if request.endpoint == "/chat/completions":
            return self.client.chat.completions.create(**request.params)
        raise ValueError(f"Unsupported test endpoint: {request.endpoint!r}")

    async def acreate(self, owner, request):
        if request.endpoint == "/responses":
            return await self.async_client.responses.create(**request.params)
        if request.endpoint == "/responses/input_tokens":
            return await self.async_client.responses.input_tokens.count(
                **request.params
            )
        if request.endpoint == "/responses/compact":
            return await self.async_client.responses.compact(**request.params)
        if request.endpoint == "/chat/completions":
            return await self.async_client.chat.completions.create(**request.params)
        raise ValueError(f"Unsupported test endpoint: {request.endpoint!r}")
