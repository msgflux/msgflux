"""httpx2 MockTransport helpers for retriever tests."""

from typing import Any, Callable, List

import httpx2


class HTTPX2Mock:
    def __init__(self, mocker, handler: Callable[[httpx2.Request], httpx2.Response]):
        self.requests: List[httpx2.Request] = []

        def recording_handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return handler(request)

        transport = httpx2.MockTransport(recording_handler)
        real_client = httpx2.Client
        real_async_client = httpx2.AsyncClient

        def client_factory(*args: Any, **kwargs: Any) -> httpx2.Client:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        def async_client_factory(*args: Any, **kwargs: Any) -> httpx2.AsyncClient:
            kwargs["transport"] = transport
            return real_async_client(*args, **kwargs)

        mocker.patch(
            "msgflux.data.retrievers.http.httpx2.Client",
            side_effect=client_factory,
        )
        mocker.patch(
            "msgflux.data.retrievers.http.httpx2.AsyncClient",
            side_effect=async_client_factory,
        )


def install_httpx2_mock(mocker, handler) -> HTTPX2Mock:
    return HTTPX2Mock(mocker, handler)


def json_response(
    request: httpx2.Request, status: int, payload: Any
) -> httpx2.Response:
    return httpx2.Response(status, json=payload, request=request)
