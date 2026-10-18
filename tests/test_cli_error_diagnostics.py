"""CLI failures keep useful text and identities without changing output modes."""

import json

import httpx2
import pytest

from msgflux import cli
from msgflux.coding.prompt import PromptExecution
from msgflux.runtime.service.local import cli as service_cli


@pytest.mark.parametrize("mode", ["plain", "json", "stream"])
@pytest.mark.parametrize("message", ["", "   ", "Connection closed"])
def test_prompt_failure_has_diagnostic_and_preserves_identities(
    monkeypatch, capsys, mode, message
):
    async def fail(self, *_args, **_kwargs):
        self.result.thread_id = "thread-1"
        self.result.request_id = "request-1"
        self.result.run_id = "run-1"
        raise httpx2.ReadTimeout(message)

    monkeypatch.setattr(PromptExecution, "run", fail)
    arguments = ["-p", "hello"]
    if mode != "plain":
        arguments.append("--json")
    if mode == "stream":
        arguments.append("--stream")
    with pytest.raises(SystemExit) as exited:
        cli.main(arguments)
    assert exited.value.code == 1
    captured = capsys.readouterr()
    expected = message if message.strip() else "ReadTimeout"
    if mode == "plain":
        assert captured.out == ""
        assert expected in captured.err
        assert "thread_id=thread-1" in captured.err
        assert "run_id=run-1" in captured.err
    else:
        assert captured.err == ""
        record = json.loads(captured.out)
        result = record["result"] if mode == "stream" else record
        assert result["error"] == expected
        assert result["status"] == "failed"
        assert result["thread_id"] == "thread-1"
        assert result["request_id"] == "request-1"
        assert result["run_id"] == "run-1"


def test_service_cli_timeout_has_diagnostic(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise httpx2.ConnectTimeout("")

    monkeypatch.setattr(service_cli, "_main", fail)
    with pytest.raises(SystemExit) as exited:
        service_cli.main(["status"])
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: ConnectTimeout\n"
