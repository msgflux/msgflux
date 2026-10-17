"""Service inspection reports the running instance's loaded HTTP policy."""

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux.runtime.service import AgentService, SQLiteServiceStore
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceHTTPError,
    ServiceHTTPConfig,
    ServiceInspection,
    create_service_app,
)


@pytest.mark.parametrize("origins", [(), ("*",), ("https://app.example",)])
@pytest.mark.asyncio
async def test_service_inspection_is_authenticated_and_typed(origins):
    service = AgentService(store=SQLiteServiceStore())
    policy = ServiceHTTPConfig(cors_allow_origins=origins)
    app = create_service_app(
        service, token="secret", http_config=policy, instance_id="test-instance"
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as transport:
            response = await transport.get("/v1/service/inspection")
            assert response.status_code == 401
            client = AgentServiceClient(
                "http://service", token="secret", client=transport
            )
            assert await client.inspect_service() == ServiceInspection(
                "test-instance", policy
            )
            assert not service.agents()
            assert not service.threads()
            assert set((await client.health()).__struct_fields__) == {
                "instance_id",
                "version",
                "default_agent",
            }
            await client.aclose()
            assert not transport.is_closed  # An injected HTTP client is borrowed.
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_inspection_of_older_server_explains_restart():
    seen = []

    async def respond(request):
        seen.append(request.url.path)
        return httpx2.Response(
            404, json={"code": "not_found", "message": "Resource not found"}
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as transport:
        client = AgentServiceClient("http://service", token="secret", client=transport)
        with pytest.raises(AgentServiceHTTPError, match="Restart or upgrade") as exc:
            await client.inspect_service()
        assert exc.value.status_code == 404
    assert seen == ["/v1/service/inspection"]


@pytest.mark.asyncio
async def test_running_inspection_does_not_reread_edited_manifest(tmp_path, capsys):
    import asyncio
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace

    from msgflux.runtime.service.local.cli import _inspect_service
    from msgflux.runtime.service.local.runner import serve_local_service
    from tests.test_local_service_runner import _wait_for_record

    if os.name != "posix":
        pytest.skip("Local runner requires POSIX lifetime locks")
    config = tmp_path / "service.toml"
    source = (
        '[defaults]\nagent = "custom"\n'
        '[agents.custom]\nfactory = "tests.test_service_cors:_session_factory"\n'
        '[http]\ncors_allow_origins = ["https://original.example"]\n'
    )
    config.write_text(source)
    runtime = tmp_path / "runtime"
    runner = asyncio.create_task(
        serve_local_service(config=config, runtime_dir=runtime, cwd=Path.cwd())
    )
    client = None
    try:
        record = await asyncio.wait_for(_wait_for_record(runtime), 5)
        client = AgentServiceClient(record.url, token=record.token)
        config.write_text(source.replace("original.example", "edited.example"))
        inspection = await client.inspect_service()
        assert inspection.instance_id == record.instance_id
        assert inspection.http.cors_allow_origins == ("https://original.example",)
        args = SimpleNamespace(runtime_dir=runtime, json=True)
        await _inspect_service(args)
        output = capsys.readouterr().out
        assert json.loads(output)["http"] == {
            "cors_allow_origins": ["https://original.example"]
        }
        assert record.token not in output
        args.json = False
        await _inspect_service(args)
        assert "CORS: https://original.example" in capsys.readouterr().out

        await client.aclose()
        client = None
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
        runner = asyncio.create_task(
            serve_local_service(config=config, runtime_dir=runtime, cwd=Path.cwd())
        )
        replacement = await asyncio.wait_for(_wait_for_record(runtime), 5)
        assert replacement.instance_id != record.instance_id
        client = AgentServiceClient(replacement.url, token=replacement.token)
        inspection = await client.inspect_service()
        assert inspection.instance_id == replacement.instance_id
        assert inspection.http.cors_allow_origins == ("https://edited.example",)
        args.json = True
        await _inspect_service(args)
        output = capsys.readouterr().out
        result = json.loads(output)
        assert result["service"]["instance_id"] == replacement.instance_id
        assert result["http"] == {"cors_allow_origins": ["https://edited.example"]}
        assert replacement.token not in output
        args.json = False
        await _inspect_service(args)
        assert "CORS: https://edited.example" in capsys.readouterr().out
    finally:
        if client is not None:
            await client.aclose()
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.parametrize("race", ["response", "record"])
@pytest.mark.asyncio
async def test_cli_inspection_rejects_identity_races_and_closes_client(
    tmp_path, monkeypatch, capsys, race
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from msgflux.runtime.service.local.cli import _inspect_service
    from msgflux.runtime.service.local.records import LocalServiceStatus

    record = SimpleNamespace(
        instance_id="original", url="http://127.0.0.1:1234", token="secret"
    )
    replacement = SimpleNamespace(instance_id="replacement")
    monkeypatch.setattr(
        "msgflux.runtime.service.local.inspect_local_service",
        AsyncMock(
            return_value=LocalServiceStatus(
                "running", str(tmp_path), instance_id="original"
            )
        ),
    )
    monkeypatch.setattr(
        "msgflux.runtime.service.local.files.read_record",
        Mock(side_effect=[record, replacement]),
    )
    client = SimpleNamespace(
        inspect_service=AsyncMock(
            return_value=ServiceInspection(
                "replacement" if race == "response" else "original", ServiceHTTPConfig()
            )
        ),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        "msgflux.runtime.service.http.client.AgentServiceClient",
        Mock(return_value=client),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        await _inspect_service(SimpleNamespace(runtime_dir=tmp_path, json=True))
    client.aclose.assert_awaited_once()
    assert not capsys.readouterr().out


@pytest.mark.asyncio
async def test_unreachable_inspection_has_no_effective_http_policy(
    tmp_path, monkeypatch, capsys
):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from msgflux.runtime.service.local.cli import _inspect_service
    from msgflux.runtime.service.local.records import LocalServiceStatus

    monkeypatch.setattr(
        "msgflux.runtime.service.local.inspect_local_service",
        AsyncMock(return_value=LocalServiceStatus("unreachable", str(tmp_path))),
    )
    await _inspect_service(SimpleNamespace(runtime_dir=tmp_path, json=True))
    assert json.loads(capsys.readouterr().out)["http"] is None
    assert not list(tmp_path.iterdir())
