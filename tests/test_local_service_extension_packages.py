"""Installed extension metadata is consumed lazily by independent daemons."""

import os
from pathlib import Path

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Local daemon uses POSIX locks"
)

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime.service.local.discovery import connect_local_service
from tests.test_local_service_integration import _shutdown_record, _wait_completed
from tests.test_local_service_manifest import clean_manifest_runtime

_PLUGIN = """
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from msgflux.nn.extensions import AgentExtension
from msgflux.nn.hooks import Hook

ROOT = Path(__file__).parent
with (ROOT / "plugin-imports").open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")


class AuditExtension(AgentExtension):
    def __init__(self, label):
        super().__init__("original-name")
        self.label = label
        self.instance_id = uuid4().hex
        with (ROOT / "plugin-instances").open("a") as stream:
            stream.write(self.instance_id + "\\n")

    def hooks(self):
        return (Hook(event="before_request", handler=self.prepare),)

    def prepare(self, context):
        state = self.durable_state()
        state["requests"] = state.get("requests", 0) + 1
        return replace(context, system_prompt=f"{self.label}:{state['requests']}")

    def on_remove(self, agent):
        with (ROOT / "plugin-removed").open("a") as stream:
            stream.write(self.instance_id + "\\n")
"""

_FACTORY = """
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
from msgflux.runtime.service import AgentSession


def create_session(thread):
    from local_session_helper import confirm_import

    confirm_import()
    root = Path(__file__).parent / thread.thread_id
    root.mkdir(exist_ok=True)
    checkpoints = SQLiteCheckpointStore(root / "checkpoints.sqlite3")
    approvals = SQLiteApprovalStore(root / "approvals.sqlite3")

    def record_effect() -> str:
        '''Append one externally visible effect.'''
        with (root / "effects").open("a") as stream:
            stream.write("effect\\n")
        return "effect recorded"

    agent = Agent(
        name=thread.agent_id, model=Mock(model_type="chat_completion"),
        tools=[record_effect], checkpoint_store=checkpoints, system_prompt="base",
        approvals=AgentApprovals(approvals, {"record_effect": "v1"}, "v1"),
    )

    async def respond(**kwargs):
        prompt = kwargs.get("system_prompt") or "base"
        with (root / "model-requests").open("a") as stream:
            stream.write(prompt + "\\n")
        response = ModelResponse()
        if thread.agent_id == "main" and not any(
            item.get("role") == "tool" for item in kwargs["messages"].to_chatml()
        ):
            calls = ToolCallAggregator()
            calls.process(0, "effect-call", "record_effect", "{}")
            response.set_response_type("tool_call")
            response.add(calls)
        else:
            response.set_response_type("text_generation")
            response.add(prompt)
        return response

    def close():
        checkpoints.close()
        approvals.close()

    agent.generator.aforward = AsyncMock(side_effect=respond)
    return AgentSession(agent, approval_reviewer="user", on_close=close)
"""


def _install_fixture_package(project: Path) -> Path:
    (project / "demo_extension.py").write_text(_PLUGIN)
    (project / "package_fixture.py").write_text(_FACTORY)
    (project / "local_session_helper.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('session-helper-imported').write_text('yes')\n"
        "def confirm_import():\n    return None\n"
    )
    config = project / "config.toml"
    config.write_text(
        "[defaults]\nagent='main'\n"
        "[extensions.demo_audit]\nfactory='demo_extension:AuditExtension'\n"
        "[extensions.demo_audit.options]\nlabel='demo'\n"
        "[agents.main]\nfactory='package_fixture:create_session'\nextensions=['demo_audit']\n"
        "[agents.plain]\nfactory='package_fixture:create_session'\n"
    )
    return config


@pytest.mark.asyncio
async def test_explicit_cwd_factories_are_lazy_and_resume_durable_hook_state(
    tmp_path, clean_manifest_runtime
):
    config = _install_fixture_package(tmp_path)
    runtime = clean_manifest_runtime
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=tmp_path
    )
    try:
        first_identity = (await client.health()).instance_id
        assert not (tmp_path / "plugin-imports").exists()
        plain = await client.open_thread("plain", thread_id="plain-thread")
        plain_admission = await client.prompt(
            plain.thread_id, "plain", request_id="plain"
        )
        assert (tmp_path / "session-helper-imported").read_text() == "yes"
        assert (
            await _wait_completed(client, plain.thread_id, "plain")
        ).status == "completed"
        assert (
            await client.run_result(plain.thread_id, plain_admission.run_id)
        ).output == "base"
        assert not (tmp_path / "plugin-imports").exists()

        thread = await client.open_thread("main", thread_id="extension-thread")
        admission = await client.prompt(
            thread.thread_id, "record an effect", request_id="one"
        )
        assert (
            await _wait_completed(client, thread.thread_id, "one")
        ).status == "paused"
        (before,) = await client.approval_reviews(thread.thread_id, admission.run_id)
        assert (
            tmp_path / "extension-thread" / "model-requests"
        ).read_text() == "demo:1\n"
        assert not (tmp_path / "extension-thread" / "effects").exists()
        first_instance = (tmp_path / "plugin-instances").read_text().splitlines()
        assert len(first_instance) == 1
    finally:
        await client.aclose()
    await _shutdown_record(runtime)
    assert (tmp_path / "plugin-removed").read_text().splitlines() == first_instance

    reopened = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=tmp_path
    )
    try:
        assert (await reopened.health()).instance_id != first_identity
        # New daemon metadata discovery has not imported or constructed the plugin.
        assert len((tmp_path / "plugin-imports").read_text().splitlines()) == 1
        assert (
            tmp_path / "plugin-instances"
        ).read_text().splitlines() == first_instance
        (review,) = await reopened.approval_reviews(thread.thread_id, admission.run_id)
        assert review == before
        instances = (tmp_path / "plugin-instances").read_text().splitlines()
        assert len(instances) == len(set(instances)) == 2
        assert len(set((tmp_path / "plugin-imports").read_text().splitlines())) == 2
        await reopened.decide_approval(
            thread.thread_id,
            admission.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )
        resumed = await reopened.resume_checkpoint(thread.thread_id, admission.run_id)
        assert resumed.run_id == admission.run_id
        completed = await _wait_completed(reopened, thread.thread_id, "one")
        assert completed.status == "completed", completed.error
        result = await reopened.run_result(thread.thread_id, admission.run_id)
        assert result.output == "demo:2"
        root = tmp_path / "extension-thread"
        assert (root / "effects").read_text() == "effect\n"
        assert (root / "model-requests").read_text().splitlines() == [
            "demo:1",
            "demo:2",
        ]
        checkpoints = SQLiteCheckpointStore(root / "checkpoints.sqlite3")
        try:
            state = checkpoints.load_state("main", thread.thread_id, admission.run_id)
            assert state["runtime"]["extensions"]["demo_audit"] == {"requests": 2}
        finally:
            checkpoints.close()
    finally:
        await reopened.aclose()
