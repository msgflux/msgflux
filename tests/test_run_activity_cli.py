"""Real CLI activity reads never start work or return conversation contents."""

import json
import os

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service.local import connect_local_service
from tests.test_local_service_integration import (
    FACTORY,
    _until,
    _wait_completed,
    daemon_workspace,
)
from tests.test_local_service_operations import _command


@pytest.mark.asyncio
async def test_activity_absent_service_does_not_start_daemon(tmp_path):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, "runs", "activity", "thread", "run")
    assert code == 1 and not out and "stopped" in err
    assert "Traceback" not in err and not runtime.exists()


@pytest.mark.asyncio
async def test_activity_cli_reads_live_then_terminal_without_new_model_calls(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await client.open_thread("main")
        receipt = await client.prompt(thread.thread_id, "block", request_id="one")
        await _until((runtime / "model_calls").exists)
        code, out, err = await _command(
            runtime, "runs", "activity", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err
        activity = json.loads(out)
        assert activity["local_worker"] and activity["status"] == "running"
        assert activity["last_event_type"] == "model.request"
        assert len(activity["recent_events"]) <= 8
        assert "block" not in out and "messages" not in out and "arguments" not in out
        code, out, err = await _command(
            runtime, "runs", "activity", thread.thread_id, receipt.run_id, as_json=False
        )
        assert code == 0 and not err and "Last observed event: model.request" in out
        assert (runtime / "model_calls").read_text().splitlines() == ["call"]
        (runtime / "release").touch()
        await _wait_completed(client, thread.thread_id, "one")
        code, out, err = await _command(
            runtime, "runs", "activity", thread.thread_id, receipt.run_id
        )
        assert code == 0 and not err
        activity = json.loads(out)
        assert activity["status"] == "completed" and not activity["local_worker"]
        assert activity["recent_events"] == [] and activity["last_event_type"] is None
        assert (runtime / "model_calls").read_text().splitlines() == ["call"]
    finally:
        (runtime / "release").touch()
        await client.aclose()
