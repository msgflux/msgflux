"""Model selection at the CLI and prompt-client boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from msgflux import cli
from msgflux.coding import prompt as prompt_module
from msgflux.coding.prompt import PromptExecution, PromptResult


@pytest.mark.parametrize("model", ["bad", "", "provider/", "/model", "provider/a b"])
def test_cli_rejects_invalid_model_before_execution(model, monkeypatch):
    constructor = AsyncMock()
    monkeypatch.setattr(prompt_module, "PromptExecution", constructor)
    with pytest.raises(SystemExit) as error:
        cli.main(["-p", "hello", "--model", model])
    assert error.value.code == 2
    constructor.assert_not_called()


@pytest.mark.asyncio
async def test_python_prompt_rejects_invalid_model_before_connect(monkeypatch):
    connect = AsyncMock()
    monkeypatch.setattr(prompt_module, "connect_local_service", connect)
    with pytest.raises(ValueError, match="provider/model-id"):
        await PromptExecution().run("hello", model="invalid")
    connect.assert_not_called()


def test_cli_passes_model_and_reports_selected_binding(monkeypatch, capsys):
    calls = []

    async def run(self, prompt, **kwargs):
        calls.append((prompt, kwargs))
        return PromptResult(
            status="accepted",
            thread_id="thread",
            agent_id="builder",
            model="provider/chosen",
            request_id="request",
            run_id="run",
        )

    monkeypatch.setattr(PromptExecution, "run", run)
    cli.main(
        [
            "-p",
            "hello",
            "--agent",
            "builder",
            "--account-name",
            "work",
            "--model",
            "provider/chosen",
            "--detach",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["model"] == "provider/chosen"
    assert calls[0][1]["model"] == "provider/chosen"
    assert calls[0][1]["account_name"] == "work"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, "provider/pinned"])
async def test_prompt_uses_model_returned_by_service_binding(model, monkeypatch):
    client = SimpleNamespace(
        open_thread=AsyncMock(
            return_value=SimpleNamespace(
                thread_id="thread",
                agent_id="builder",
                model="provider/pinned",
            )
        ),
        prompt=AsyncMock(
            return_value=SimpleNamespace(
                run_id="run",
                status="accepted",
                error=None,
            )
        ),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        prompt_module, "connect_local_service", AsyncMock(return_value=client)
    )
    result = await PromptExecution().run("hello", model=model, detach=True)
    assert result.model == "provider/pinned"
    assert client.open_thread.await_args.kwargs["model"] == model
    client.aclose.assert_awaited_once()


def test_text_admission_reports_bound_model(capsys):
    cli._write_identities(
        PromptResult(
            thread_id="thread",
            model="provider/pinned",
            run_id="run",
        )
    )
    assert "model=provider/pinned" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_coding_factory_uses_thread_model_without_changing_child_model(
    tmp_path, monkeypatch
):
    from msgflux.coding import factory as factory_module
    from msgflux.coding.config import CodingAgentCatalog, coding_agent_options
    from msgflux.runtime.service.records import ServiceThread
    from tests.test_coding_factory import _FakeModel

    created = []

    def create(options):
        model = _FakeModel()
        created.append((options, model))
        return model

    monkeypatch.setattr(factory_module, "_create_model", create)
    parent = coding_agent_options(
        "coding", "provider/original", active=("read",), subagents=("explorer",)
    )
    child = coding_agent_options("explorer", "provider/child", active=("read",))
    build = factory_module.create_coding_session_factory(
        options=parent,
        catalog=CodingAgentCatalog(agents={"coding": parent, "explorer": child}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
    )
    session = await build(
        ServiceThread("thread", "coding", str(tmp_path), False, "other/selected")
    )
    try:
        assert [options.model for options, _ in created] == [
            "other/selected",
            "provider/child",
        ]
        assert created[0][0].tools_active == parent.tools_active
        assert parent.model == "provider/original"
    finally:
        await session.on_close()
    assert all(model.closed for _, model in created)
