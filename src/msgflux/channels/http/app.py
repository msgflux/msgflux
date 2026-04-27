from importlib import import_module
from typing import Any

import msgspec

from msgflux.channels.exceptions import ChannelError
from msgflux.channels.http.msgspec import make_msgspec_classes
from msgflux.channels.http.openai import (
    create_chat_completion,
    create_chat_completion_stream,
    decode_chat_completion_request,
    encode_error,
    encode_json,
)
from msgflux.channels.registry import ChannelRegistry


def create_app(registry: ChannelRegistry, **fastapi_kwargs: Any):
    try:
        fastapi_cls = import_module("fastapi").FastAPI
        request_cls = import_module("fastapi").Request
        responses = import_module("fastapi.responses")
        response_cls = responses.Response
        streaming_response_cls = responses.StreamingResponse
    except ImportError as e:
        raise ImportError(
            "The msgflux server requires FastAPI. Install it with "
            "`pip install msgflux[server]`."
        ) from e

    msgspec_json_response, _, msgspec_route = make_msgspec_classes()
    fastapi_kwargs.setdefault("default_response_class", msgspec_json_response)
    app = fastapi_cls(**fastapi_kwargs)
    app.router.route_class = msgspec_route

    @app.get("/health")
    async def health():
        return {"status": "ok", "agents": sorted(registry.agents())}

    @app.post("/v1/chat/completions", response_class=msgspec_json_response)
    async def chat_completions(http_request: request_cls):
        try:
            request = decode_chat_completion_request(await http_request.body())
        except msgspec.ValidationError as e:
            return response_cls(
                content=encode_error(str(e), code="invalid_request"),
                status_code=400,
                media_type="application/json",
            )

        try:
            if request.stream:
                registry.get_agent(request.model)
                return streaming_response_cls(
                    create_chat_completion_stream(registry, request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )

            response = await create_chat_completion(registry, request)
            return response_cls(
                content=encode_json(response),
                media_type="application/json",
            )
        except ChannelError as e:
            return response_cls(
                content=encode_error(e.message, code=e.code),
                status_code=e.status_code,
                media_type="application/json",
            )

    return app
