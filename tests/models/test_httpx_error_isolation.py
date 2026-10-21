"""Behavioral checks for legacy model HTTP error and header handling."""

import asyncio

import httpx2
import pytest

from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.httpx import HTTPXModelClient
from msgflux.models.model_credentials import ResolvedModelCredentials


class _CredentialResolver:
    @staticmethod
    def resolve(owner):
        return ResolvedModelCredentials(headers={})


class _Owner:
    provider = "test-provider"
    model_id = "test-model"
    sampling_params = {"base_url": "https://api.example.test/v1"}
    credential_resolver = _CredentialResolver()

    @staticmethod
    def _raise_if_aborted():
        return None


class _ConcreteHTTPXClient(HTTPXModelClient):
    provider = "test-provider"
    model_id = "test-model"
    endpoint = "/chat"
    sampling_params = {"base_url": "https://api.example.test/v1"}

    def __call__(self, **kwargs):
        return self._execute(**kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_legacy_httpx_model_http_failure_is_normalized(asynchronous):
    def handler(request):
        return httpx2.Response(
            400,
            json={"error": {"message": "Bad request", "code": "bad_input"}},
        )

    sync_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    model.client = sync_client
    model.aclient = async_client

    try:
        if asynchronous:
            with pytest.raises(ModelProviderHTTPError) as raised:
                await model._aexecute()
        else:
            with pytest.raises(ModelProviderHTTPError) as raised:
                model._execute()

        assert raised.value.status_code == 400
        assert raised.value.description == "Bad request"
        assert raised.value.code == "bad_input"
        assert isinstance(raised.value.__cause__, httpx2.HTTPStatusError)
    finally:
        sync_client.close()
        await async_client.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_legacy_httpx_model_success_response_is_unchanged(asynchronous):
    def handler(request):
        return httpx2.Response(200, json={"result": "ok"})

    async def async_handler(request):
        return httpx2.Response(200, json={"result": "ok"})

    sync_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
    model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    model.client = sync_client
    model.aclient = async_client

    try:
        if asynchronous:
            assert await model._aexecute() == {"result": "ok"}
        else:
            assert model._execute() == {"result": "ok"}
    finally:
        sync_client.close()
        await async_client.aclose()


def test_legacy_httpx_instances_do_not_share_authorization_headers():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"result": "ok"})

    sync_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    keyed_model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    keyed_model.client = sync_client
    keyed_model._api_key = "test-secret"
    keyless_model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    keyless_model.client = sync_client

    try:
        keyed_model._execute()
        keyless_model._execute()

        assert captured[0].headers["authorization"] == "Bearer test-secret"
        assert "authorization" not in captured[1].headers
    finally:
        sync_client.close()


def test_shared_http_transport_normalizes_provider_http_failure():
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                400,
                json={"error": {"message": "Bad request", "code": "bad_input"}},
            )
        )
    )
    transport = HTTPTransport(client=client, max_retries=0)

    try:
        with pytest.raises(ModelProviderHTTPError) as raised:
            transport.request(_Owner(), "/chat")

        assert raised.value.description == "Bad request"
        assert raised.value.code == "bad_input"
        assert isinstance(raised.value.__cause__, httpx2.HTTPStatusError)
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_legacy_httpx_instances_do_not_share_authorization_headers():
    captured = []

    async def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"result": "ok"})

    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    keyed_model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    keyed_model.aclient = async_client
    keyed_model._api_key = "test-secret"
    keyless_model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
    keyless_model.aclient = async_client

    try:
        await keyed_model._aexecute()
        await keyless_model._aexecute()

        assert captured[0].headers["authorization"] == "Bearer test-secret"
        assert "authorization" not in captured[1].headers
    finally:
        await async_client.aclose()


@pytest.mark.asyncio
async def test_concurrent_legacy_httpx_instances_keep_their_own_authorization():
    captured = []

    async def handler(request):
        captured.append(request.headers["authorization"])
        return httpx2.Response(200, json={"result": "ok"})

    async_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    models = []
    for key in ("first-secret", "second-secret"):
        model = _ConcreteHTTPXClient.__new__(_ConcreteHTTPXClient)
        model.aclient = async_client
        model._api_key = key
        models.append(model)

    try:
        await asyncio.gather(*(model._aexecute() for model in models))

        assert set(captured) == {"Bearer first-secret", "Bearer second-secret"}
    finally:
        await async_client.aclose()
