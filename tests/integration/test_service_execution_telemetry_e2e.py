"""Opt-in Codex + TCP AgentService + Docker OpenTelemetry Collector validation.

MSGFLUX_LIVE_SERVICE_TELEMETRY=1 uv run --extra service pytest -q \
    tests/integration/test_service_execution_telemetry_e2e.py

Uses the existing ``work`` account (override MSGFLUX_LIVE_SERVICE_ACCOUNT).
Runs the instrumented application in a subprocess to exercise actual global
provider configuration without contaminating the offline test process.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
from uuid import uuid4

import pytest


_IMAGE = "otel/opentelemetry-collector-contrib:0.163.1"


def _docker(*args, check=True):
    return subprocess.run(  # noqa: S603
        [shutil.which("docker") or "/usr/bin/docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def _exported_spans(path):
    spans = []
    for line in path.read_text().splitlines():
        for resource in json.loads(line).get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                spans.extend(scope.get("spans", []))
    return spans


def _attributes(span):
    return {
        item["key"]: next(iter(item["value"].values()))
        for item in span.get("attributes", [])
    }


@pytest.mark.skipif(
    os.getenv("MSGFLUX_LIVE_SERVICE_TELEMETRY") != "1",
    reason="Set MSGFLUX_LIVE_SERVICE_TELEMETRY=1 for live Codex/OTLP checks",
)
def test_real_codex_attempts_export_links_models_tools_and_resumption(tmp_path):
    # Mount a dedicated directory; the collector's non-root user owns its output.
    collector_dir = tmp_path / "collector"
    collector_dir.mkdir(mode=0o777)
    collector_dir.chmod(0o777)
    config = collector_dir / "config.yaml"
    config.write_text(
        "receivers:\n  otlp:\n    protocols:\n      http:\n"
        "        endpoint: 0.0.0.0:4318\n"
        "exporters:\n  file:\n    path: /capture/traces.jsonl\n"
        "    flush_interval: 100ms\n"
        "service:\n  pipelines:\n    traces:\n"
        "      receivers: [otlp]\n      exporters: [file]\n"
    )
    container = f"msgflux-telemetry-{uuid4().hex[:12]}"
    try:
        _docker(
            "run",
            "--detach",
            "--name",
            container,
            "--publish",
            "127.0.0.1::4318",
            "--volume",
            f"{collector_dir}:/capture",
            _IMAGE,
            "--config=/capture/config.yaml",
        )
        address = _docker("port", container, "4318/tcp")
        host, port = address.rsplit(":", 1)
        deadline = time.monotonic() + 15
        while True:
            try:
                with socket.create_connection((host, int(port)), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    pytest.fail("Collector did not become ready")
                time.sleep(0.1)
        env = dict(os.environ)
        env.update(
            {
                "MSGTRACE_TELEMETRY_ENABLED": "true",
                "MSGTRACE_EXPORTER": "otlp",
                "MSGTRACE_OTLP_ENDPOINT": f"http://{address}/v1/traces",
                "MSGTRACE_SERVICE_NAME": "msgflux-live-telemetry",
            }
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, __file__, str(tmp_path)],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        expected = json.loads((tmp_path / "expected.json").read_text())
        output = collector_dir / "traces.jsonl"
        deadline = time.monotonic() + 10
        while True:
            spans = _exported_spans(output) if output.exists() else []
            attempts = [s for s in spans if s["name"] == "msgflux.service.execute"]
            if len(attempts) == 3:
                break
            assert time.monotonic() < deadline, (
                "Collector did not export three attempts"
            )
            time.sleep(0.1)
        statuses = sorted(_attributes(s)["msgflux.run.status"] for s in attempts)
        assert statuses == ["completed", "completed", "paused"]
        assert len({s["traceId"] for s in attempts}) == 3
        for span in attempts:
            attrs = _attributes(span)
            thread_id = attrs["msgflux.thread_id"]
            assert attrs["msgflux.run_id"] == expected[thread_id]
            assert not span.get("parentSpanId")
            assert len(span["links"]) == 1
            link = span["links"][0]
            parent = next(
                parent
                for parent in spans
                if parent["spanId"] == link["spanId"]
                and parent["traceId"] == link["traceId"]
            )
            resumed_attempt = (
                attrs["msgflux.agent_id"] == "approved"
                and attrs["msgflux.run.status"] == "completed"
            )
            assert parent["name"].endswith("/resume" if resumed_attempt else "/prompt")
            descendants = [
                s for s in spans if s["traceId"] == span["traceId"] and s != span
            ]
            assert descendants, "Attempt must contain actual execution spans"
            assert any("gen_ai.request.model" in _attributes(s) for s in descendants)
            span_ids = {span["spanId"], *(s["spanId"] for s in descendants)}
            assert all(s.get("parentSpanId") in span_ids for s in descendants)
        resumed = next(
            s
            for s in attempts
            if _attributes(s)["msgflux.agent_id"] == "approved"
            and _attributes(s)["msgflux.run.status"] == "completed"
        )
        assert any(
            s["traceId"] == resumed["traceId"]
            and _attributes(s).get("gen_ai.tool.name") == "lookup"
            for s in spans
        ), "Approved tool must belong to the resumed execution trace"
        watcher = next(s for s in spans if s["name"].endswith("/watch"))
        assert int(watcher["endTimeUnixNano"]) < int(resumed["endTimeUnixNano"])
        # New service spans contain metadata only; existing model/tool capture
        # settings remain independently configurable.
        assert all(not s.get("events") for s in attempts)
    finally:
        _docker("rm", "--force", container, check=False)


async def _exercise(directory: Path):
    import httpx2
    import uvicorn
    from opentelemetry import trace

    import msgflux as mf
    from msgflux.accounts import Accounts
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.envs import envs
    from msgflux.models.providers.openai_codex import CodexChatTransport
    from msgflux.nn import Agent
    from msgflux.runtime import AgentApprovals, InMemoryApprovalStore
    from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
    from msgflux.runtime.service.http import create_service_app
    from msgflux.runtime.service.http.client import AgentServiceClient

    envs.telemetry_capture_model_output = False
    envs.telemetry_capture_tool_call_responses = False
    calls = []
    checkpoints = []
    tool_release = asyncio.Event()
    tool_started = asyncio.Event()
    account = os.getenv("MSGFLUX_LIVE_SERVICE_ACCOUNT", "work")

    async def lookup(code: str) -> str:
        """Look up the verification value for an archive code."""
        calls.append(code)
        tool_started.set()
        await asyncio.wait_for(tool_release.wait(), 10)
        return "TRACE-READY"

    def factory(thread):
        model = mf.Model.chat_completion(
            "openai-codex/" + os.getenv("MSGFLUX_LIVE_CODEX_MODEL", "gpt-6-luna"),
            api_mode="responses",
            reasoning_effort="medium",
            retry=False,
            credential_resolver=Accounts().resolver(
                "openai-codex", account_name=account
            ),
            chat_transport=CodexChatTransport(timeout=120, max_retries=0),
        )
        store = SQLiteCheckpointStore(directory / f"{thread.thread_id}.sqlite3")
        checkpoints.append(store)
        approved = thread.agent_id == "approved"
        agent = Agent(
            name=thread.agent_id,
            model=model,
            checkpoint_store=store,
            tools=[lookup] if approved else [],
            approvals=AgentApprovals(InMemoryApprovalStore(), {"lookup": "v1"}, "v1")
            if approved
            else None,
            system_prompt="Follow the user's instructions exactly. Keep the final response short.",
        )
        return AgentSession(agent, checkpoint_store=store, approval_reviewer="operator")

    service = AgentService(
        store=SQLiteServiceStore(directory / "service.sqlite3"), default_account=account
    )
    service.register("approved", factory)
    service.register("plain", factory)
    app = create_service_app(
        service, token="live-test-token", instance_id="live-telemetry"
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        address = f"http://127.0.0.1:{sock.getsockname()[1]}"
        async with httpx2.AsyncClient(timeout=120) as http:
            client = AgentServiceClient(address, token="live-test-token", client=http)
            approved = await client.open_thread("approved")
            plain = await client.open_thread("plain")
            http.headers["traceparent"] = (
                "00-11111111111111111111111111111111-1111111111111111-01"
            )
            first, other = await asyncio.gather(
                client.prompt(
                    approved.thread_id,
                    "Use lookup exactly once with code R-17. Then answer with its returned value only.",
                    request_id="approved",
                ),
                client.prompt(
                    plain.thread_id,
                    "Reply with READY only. Do not use tools.",
                    request_id="plain",
                ),
            )
            paused, done = await asyncio.wait_for(
                asyncio.gather(
                    service.wait(approved.thread_id, first.request_id),
                    service.wait(plain.thread_id, other.request_id),
                ),
                150,
            )
            assert paused.status == "paused", paused.error
            assert done.status == "completed", done.error
            assert calls == []
            (review,) = await client.approval_reviews(approved.thread_id, first.run_id)
            await client.decide_approval(
                approved.thread_id,
                first.run_id,
                review.request_id,
                approved=True,
                expected_revision=review.revision,
            )
            http.headers["traceparent"] = (
                "00-22222222222222222222222222222222-2222222222222222-01"
            )
            # The watcher disconnects while the approved tool is still running.
            async with client.watch(approved.thread_id):
                await client.resume_checkpoint(approved.thread_id, first.run_id)
                await asyncio.wait_for(tool_started.wait(), 10)
            assert await client.active_runs()
            tool_release.set()
            completed = await asyncio.wait_for(
                service.wait(approved.thread_id, first.request_id), 150
            )
            assert completed.status == "completed", completed.error
            final = await client.run_result(approved.thread_id, first.run_id)
            assert "TRACE-READY" in final.output
            assert calls == ["R-17"]
            assert not await client.active_runs()
            (directory / "expected.json").write_text(
                json.dumps(
                    {approved.thread_id: first.run_id, plain.thread_id: other.run_id}
                )
            )
    finally:
        tool_release.set()
        server.should_exit = True
        await asyncio.wait_for(task, 10)
        sock.close()
        await service.aclose()
        service.store.close()
        for store in checkpoints:
            store.close()
        provider = trace.get_tracer_provider()
        assert provider.force_flush(timeout_millis=10000)
        provider.shutdown()


if __name__ == "__main__":
    asyncio.run(_exercise(Path(sys.argv[1])))
