"""Typed inspection over HTTP and a real daemon without constructing models."""

import json
import os
from pathlib import Path

import httpx2
import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

from msgflux.runtime.service import AgentService, SQLiteServiceStore
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceHTTPError,
    create_service_app,
)
from msgflux.runtime.service.local import connect_local_service, stop_local_service
from msgflux.runtime.service.local.cli import main
from tests.test_local_service_operations import _command


@pytest.mark.asyncio
async def test_http_unknown_agent_and_auth_never_call_factory():
    service = AgentService(store=SQLiteServiceStore())

    def unexpected(_thread):
        raise AssertionError("Factory must remain lazy")

    service.register("custom", unexpected)
    app = create_service_app(service, token="secret")
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.get("/v1/agents/custom/inspection")
            assert response.status_code == 401
            client = AgentServiceClient("http://test", token="secret", client=http)
            result = await client.inspect_agent("custom")
            assert result.configuration is None and not result.is_default
            with pytest.raises(AgentServiceHTTPError, match="not registered") as raised:
                await client.inspect_agent("missing")
            assert raised.value.status_code == 404
            assert service.threads() == ()
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_missing_endpoint_diagnostic_is_distinct_from_unknown_agent():
    async def handler(request):
        if request.url.path == "/v1/agents":
            return httpx2.Response(200, json={"agents": ["known"]}, request=request)
        return httpx2.Response(
            404, json={"code": "not_found", "message": "not found"}, request=request
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://old", token="secret", client=http)
        with pytest.raises(AgentServiceHTTPError, match="Restart or upgrade"):
            await client.inspect_agent("known")
        with pytest.raises(AgentServiceHTTPError, match=r"^not found$"):
            await client.inspect_agent("missing")


@pytest.mark.parametrize("args", [["inspect"], ["inspect", "coding", "extra"]])
def test_agent_inspection_requires_exactly_one_agent_id(args):
    with pytest.raises(SystemExit, match="exactly one AGENT_ID"):
        main(args, resource="agents")


@pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")
@pytest.mark.asyncio
async def test_cli_absent_service_does_not_start_or_create_runtime(tmp_path):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, "agents", "inspect", "coding")
    assert code == 1 and not out and "stopped" in err
    assert not runtime.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")
@pytest.mark.asyncio
async def test_real_daemon_inspects_loaded_defaults_without_credentials_or_sessions(
    tmp_path,
):
    config = tmp_path / "config.toml"
    runtime = tmp_path / "runtime"
    prompt = tmp_path / "system.md"
    prompt.write_text("PRIVATE_PROMPT_CONTENT_MUST_NOT_BE_EXPOSED")
    config.write_text("""[defaults]
agent = "coding"
model = "openai-codex/unavailable-model"
account = "not-logged-in"
accounts_dir = "missing-accounts"
agent_dir = "agent-state"
[agents.coding]
harness = "coding"
reasoning_effort = "high"
prompt_file = "system.md"
subagents = ["explorer"]
[agents.coding.tools]
deferred = ["web_fetch"]
[agents.explorer]
harness = "coding"
preset = "explorer"
""")
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=tmp_path
    )
    try:
        loaded = await client.inspect_agent("coding")
        assert loaded.is_default and loaded.account_name == "not-logged-in"
        metadata = loaded.configuration
        assert metadata.model == "openai-codex/unavailable-model"
        assert metadata.reasoning_effort == "high"
        assert metadata.tools_active == ("bash", "read", "apply_patch")
        assert metadata.tools_deferred == ("web_fetch",) and metadata.subagents == (
            "explorer",
        )
        assert metadata.prompt_source == "file" and metadata.prompt_file == str(prompt)
        config.write_text("invalid config after startup")
        prompt.unlink()
        # This must inspect the server registration, not load the changed config/prompt.
        code, out, err = await _command(runtime, "agents", "inspect", "coding")
        assert code == 0 and not err
        result = json.loads(out)
        assert result["configuration"]["config_digest"] == metadata.config_digest
        assert result["configuration"]["reasoning_effort"] == "high"
        assert "PRIVATE_PROMPT" not in out and "token" not in out
        code, out, err = await _command(
            runtime, "agents", "inspect", "explorer", as_json=False
        )
        assert code == 0 and not err and "Prompt source: preset" in out
        assert (
            "Permissions: read" in out
            and "Tools (active defaults): read, ls, glob, grep" in out
        )
        code, out, err = await _command(runtime, "agents", "inspect", "missing")
        assert (
            code == 1 and not out and "not registered" in err and "Restart" not in err
        )
        assert await client.threads() == () and await client.active_runs() == ()
        assert not (runtime / "workspaces.sqlite3").exists()
        assert not (tmp_path / "missing-accounts").exists()
        assert not (tmp_path / "agent-state").exists()
        assert not list(tmp_path.rglob("checkpoints.sqlite3"))
    finally:
        await client.aclose()
        await stop_local_service(runtime_dir=runtime)
