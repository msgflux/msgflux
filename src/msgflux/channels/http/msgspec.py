from importlib import import_module
from typing import Any, Callable

import msgspec


def make_msgspec_classes():
    try:
        api_route_cls = import_module("fastapi.routing").APIRoute
        json_response_cls = import_module("fastapi.responses").JSONResponse
        request_cls = import_module("starlette.requests").Request
    except ImportError as e:
        raise ImportError(
            "The msgflux server requires FastAPI. Install it with "
            "`pip install msgflux[server]`."
        ) from e

    class MsgspecJSONResponse(json_response_cls):
        def render(self, content: Any) -> bytes:
            return msgspec.json.encode(content)

    class MsgspecJSONRequest(request_cls):
        async def json(self) -> Any:
            if not hasattr(self, "_json"):
                self._json = msgspec.json.decode(await self.body())
            return self._json

    class MsgspecRoute(api_route_cls):
        def get_route_handler(self) -> Callable[[Any], Any]:
            original_route_handler = super().get_route_handler()

            async def custom_route_handler(request: Any) -> Any:
                request = MsgspecJSONRequest(request.scope, request.receive)
                return await original_route_handler(request)

            return custom_route_handler

    return MsgspecJSONResponse, MsgspecJSONRequest, MsgspecRoute
