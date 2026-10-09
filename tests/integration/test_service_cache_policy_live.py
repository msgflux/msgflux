"""Opt-in live stress of cache eviction with real Codex calls and SQLite tools.

Enable with ``MSGFLUX_LIVE_SERVICE_CACHE_POLICY=1`` and valid Codex CLI auth.
The artifact contains resource metrics and task/run statuses only; it never
contains prompts, model responses, tool output, file contents, or credentials.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import AsyncExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import msgflux as mf
from msgflux.models.providers.openai_codex import CodexChatTransport
from msgflux.nn import Agent, ToolTurnLimitExtension
from msgflux.nn.extensions import ManagedToolOutputOffloadExtension
from msgflux.runtime import ToolOutputOffloadConfig
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    SessionCachePolicy,
    SQLiteServiceStore,
)
from msgflux.tools.builtin import AgentTool, BashTool, ReadFileTool


_OPT_IN = "MSGFLUX_LIVE_SERVICE_CACHE_POLICY"
_THREAD_IDS = ("cache-live-a", "cache-live-b", "cache-live-c")
_PAYLOAD_SIZE = 24 * 1024 * 1024
_NOISE_MARKER = "CACHE_POLICY_LIVE_TOOL_OUTPUT_MARKER_"


def _write_payload(path: Path) -> None:
    line = b"msgflux-service-cache-policy-live-payload-0123456789\n"
    count, remainder = divmod(_PAYLOAD_SIZE, len(line))
    with path.open("wb") as stream:
        block = line * 4096
        for _ in range(count // 4096):
            stream.write(block)
        stream.write(line * (count % 4096))
        stream.write(line[:remainder])


def _child_system_prompt() -> str:
    return (
        "Inspect the beginning of payload.bin with the read tool, requesting at "
        "most 24 lines. Then use bash to run this exact command: "
        '`sleep 5; python3 -c \'import hashlib; p=open("payload.bin","rb").read(); '
        'print("BYTES="+str(len(p))); '
        'print("SHA256="+hashlib.sha256(p).hexdigest()); '
        f'print("{_NOISE_MARKER}"*6000)\'`. '
        "Report only the SHA-256 and byte count. Do not quote file contents or "
        "the marker output."
    )


def _safe_process_sample(psutil, service):
    process = psutil.Process()
    children = process.children(recursive=True)
    child_rss = 0
    child_fds = 0
    live_children = 0
    for child in children:
        try:
            child_rss += child.memory_info().rss
            live_children += 1
            if hasattr(child, "num_fds"):
                child_fds += child.num_fds()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    try:
        parent_fds = process.num_fds()
    except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess):
        parent_fds = None
    parent_rss = process.memory_info().rss
    return {
        "timestamp": time.monotonic(),
        "server_rss_bytes": parent_rss,
        "child_rss_bytes": child_rss,
        "aggregate_rss_bytes": parent_rss + child_rss,
        "server_fd_count": parent_fds,
        "child_fd_count": child_fds,
        "child_process_count": live_children,
        "loaded_sessions": len(service._cache.sessions),
        "owned_bundles_open": sum(
            not owned.resources._closed
            for session in service._cache.sessions.values()
            for owned in session.agent._owned_threads.values()
        ),
    }


def _sample_sqlite(database_paths, previous_rows):
    sample = {
        "sqlite_read_queries": 0,
        "sqlite_busy_errors": 0,
        "sqlite_state_changes_observed": 0,
        "sqlite_store_bytes": 0,
        "sqlite_wal_bytes": 0,
        "sqlite_pending_tasks": 0,
        "sqlite_running_tasks": 0,
    }
    for path in database_paths:
        if not path.is_file():
            continue
        sample["sqlite_store_bytes"] += path.stat().st_size
        wal_path = Path(str(path) + "-wal")
        if wal_path.exists():
            sample["sqlite_wal_bytes"] += wal_path.stat().st_size
        try:
            connection = sqlite3.connect(path, timeout=0.05)
            try:
                rows = connection.execute(
                    "SELECT task_id, status, updated_at FROM tasks"
                ).fetchall()
            finally:
                connection.close()
            sample["sqlite_read_queries"] += 1
            sample["sqlite_pending_tasks"] += sum(
                status in {"queued", "running"} for _task_id, status, _updated in rows
            )
            sample["sqlite_running_tasks"] += sum(
                status == "running" for _task_id, status, _updated in rows
            )
            current = tuple(rows)
            if path in previous_rows and previous_rows[path] != current:
                sample["sqlite_state_changes_observed"] += 1
            previous_rows[path] = current
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                sample["sqlite_busy_errors"] += 1
            else:
                raise
    return sample


async def _sample_processes(psutil, service, database_paths, stop, samples):
    previous_rows = {}
    while not stop.is_set():
        try:
            sample = _safe_process_sample(psutil, service)
            sample.update(_sample_sqlite(database_paths, previous_rows))
            samples.append(sample)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            continue


async def _wait_for_live_tasks(service, threads, receipts, *, timeout=240):
    deadline = asyncio.get_running_loop().time() + timeout
    peak_pending = 0
    peak_running = 0
    peak_summaries = {}
    latest = {}
    root_waiters = {
        thread_id: asyncio.create_task(
            service.wait(thread_id, receipts[thread_id].request_id)
        )
        for thread_id in threads
    }
    try:
        while True:
            latest = dict(
                zip(
                    threads,
                    await asyncio.gather(
                        *(
                            service.inspect_run(thread_id, receipts[thread_id].run_id)
                            for thread_id in threads
                        )
                    ),
                )
            )
            pending = sum(
                task.status in {"queued", "running"}
                for inspection in latest.values()
                for task in inspection.background_tasks
            )
            running = sum(
                task.status == "running"
                for inspection in latest.values()
                for task in inspection.background_tasks
            )
            if pending > peak_pending:
                peak_pending = pending
                peak_summaries = {
                    thread_id: [
                        {
                            "tool_name": task.tool_name,
                            "status": task.status,
                        }
                        for task in inspection.background_tasks
                    ]
                    for thread_id, inspection in latest.items()
                }
            peak_running = max(peak_running, running)

            roots_settled = all(task.done() for task in root_waiters.values())
            failed_children = [
                (thread_id, task.tool_name, task.status)
                for thread_id, inspection in latest.items()
                for task in inspection.background_tasks
                if task.status in {"failed", "interrupted", "paused"}
            ]
            if failed_children:
                raise AssertionError(
                    f"Live background child tasks did not complete: {failed_children}"
                )
            if roots_settled and any(
                len(inspection.background_tasks) != 2 for inspection in latest.values()
            ):
                raise AssertionError(
                    "A completed root did not persist both background children"
                )
            child_tasks_settled = all(
                len(inspection.background_tasks) == 2
                and all(
                    task.status == "completed" for task in inspection.background_tasks
                )
                for inspection in latest.values()
            )
            if roots_settled and child_tasks_settled:
                receipts_after_wait = await asyncio.gather(*root_waiters.values())
                return (
                    latest,
                    receipts_after_wait,
                    peak_running,
                    peak_pending,
                    peak_summaries,
                )
            if asyncio.get_running_loop().time() >= deadline:
                statuses = {
                    thread_id: [
                        (task.tool_name, task.status)
                        for task in inspection.background_tasks
                    ]
                    for thread_id, inspection in latest.items()
                }
                raise TimeoutError(
                    "Live child tasks did not settle within the bounded window: "
                    f"{statuses}"
                )
            await asyncio.sleep(0.05)
    finally:
        for task in root_waiters.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*root_waiters.values(), return_exceptions=True)


def _store_integrity(path: Path) -> str:
    assert path.is_file()
    connection = sqlite3.connect(path, timeout=5)
    try:
        return connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()


def _validate_inspections(inspections, receipts):
    for thread_id, inspection in inspections.items():
        assert inspection.receipt is not None
        assert inspection.receipt.status == "completed"
        assert len(inspection.background_tasks) == 2
        assert all(
            task.tool_name == "agent"
            and task.status == "completed"
            and task.error is None
            for task in inspection.background_tasks
        )
        assert inspection.receipt.run_id == receipts[thread_id].run_id


def _validate_closed_stores(
    managed_root, receipts, inspections, tools_config, workspace_roots
):
    integrity = {}
    offloaded_counts = {}
    database_metadata = {}
    for thread_id, receipt in receipts.items():
        state_thread = managed_root / "threads" / thread_id
        integrity[thread_id] = {
            name: _store_integrity(state_thread / filename)
            for name, filename in (
                ("checkpoints", "checkpoints.sqlite3"),
                ("tasks", "tasks.sqlite3"),
                ("inbox", "inbox.sqlite3"),
                ("approvals", "approvals.sqlite3"),
            )
        }
        assert set(integrity[thread_id].values()) == {"ok"}
        task_database = sqlite3.connect(state_thread / "tasks.sqlite3")
        try:
            rows = task_database.execute(
                "SELECT task_id, status, error, metadata, result FROM tasks"
            ).fetchall()
        finally:
            task_database.close()
        assert len(rows) == 2
        expected_ids = {
            task.task_id for task in inspections[thread_id].background_tasks
        }
        assert {row[0] for row in rows} == expected_ids
        namespaces = set()
        expected_payload = workspace_roots[thread_id] / "payload.bin"
        expected_bytes = expected_payload.stat().st_size
        with expected_payload.open("rb") as payload_stream:
            expected_sha256 = hashlib.file_digest(payload_stream, "sha256").hexdigest()
        result_summaries = []
        for _task_id, status, error, raw_metadata, raw_result in rows:
            metadata = json.loads(raw_metadata)
            namespaces.add(metadata.get("checkpoint_namespace"))
            assert status == "completed"
            assert error is None
            assert metadata.get("thread_id") == thread_id
            assert metadata.get("root_run_id") == receipt.run_id
            result = json.loads(raw_result)
            assert isinstance(result, str)
            assert "ModelStreamResponse" not in result
            assert expected_sha256 in result.lower()
            normalized_result = re.sub(r"[\s,_]", "", result)
            assert str(expected_bytes) in normalized_result
            result_summaries.append(
                {
                    "result_type": type(result).__name__,
                    "result_size_bytes": len(result.encode("utf-8")),
                }
            )
        assert namespaces == {"io_a", "io_b"}
        database_metadata[thread_id] = {
            "task_count": len(rows),
            "namespaces": sorted(namespaces),
            "results": result_summaries,
        }

        artifacts = list((state_thread / "tool-results").glob("*/content"))
        large_artifacts = [path for path in artifacts if path.stat().st_size > 100_000]
        # Identical output content may be deduplicated while each task keeps
        # its own result reference in the durable task record.
        assert large_artifacts
        assert all(
            path.stat().st_size <= tools_config.max_result_bytes
            for path in large_artifacts
        )
        marker = _NOISE_MARKER.encode()
        assert any(marker in path.read_bytes() for path in large_artifacts)
        offloaded_counts[thread_id] = len(large_artifacts)
    return integrity, database_metadata, offloaded_counts


async def _drain_until_runs(
    watcher, expected_run_ids, *, timeout=30, quiet_period=0.25
):
    events = []
    terminal_events = []
    observed_run_ids = set()
    iterator = watcher.__aiter__()
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Watcher did not deliver all expected run terminals")
        wait = quiet_period if expected_run_ids <= observed_run_ids else remaining
        try:
            event = await asyncio.wait_for(
                anext(iterator), timeout=min(wait, remaining)
            )
        except TimeoutError:
            if expected_run_ids <= observed_run_ids:
                break
            raise
        except StopAsyncIteration:
            break
        events.append(event)
        if event.type in {"run.end", "run.error", "run.interrupted", "run.paused"}:
            terminal_events.append(event)
            if event.run_id in expected_run_ids:
                observed_run_ids.add(event.run_id)
    return events, terminal_events


async def _drain_watchers(watchers, receipts, inspections):
    summary = {}
    for thread_id, watcher in watchers.items():
        child_run_ids = {
            task.task_id for task in inspections[thread_id].background_tasks
        }
        expected_run_ids = {receipts[thread_id].run_id, *child_run_ids}
        events, terminal_events = await _drain_until_runs(watcher, expected_run_ids)
        summary[thread_id] = _summarize_watcher_events(
            watcher, events, expected_run_ids, terminal_events, child_run_ids
        )
    return summary


def _summarize_watcher_events(
    watcher, events, expected_run_ids, terminal_events, child_run_ids
):
    issues = []
    terminal_run_ids = {event.run_id for event in terminal_events}
    if expected_run_ids - terminal_run_ids:
        issues.append("missing_run_terminals")
    if any(
        event.type != "run.end" and event.run_id in expected_run_ids
        for event in terminal_events
    ):
        issues.append("failed_or_interrupted_run")
    if any(event.type == "tool.end" and event.data.get("error") for event in events):
        issues.append("tool_error")
    tool_events, child_tool_issues = _count_child_tool_events(events, child_run_ids)
    issues.extend(child_tool_issues)
    output_descriptors = _summarize_tool_output_events(events)
    if output_descriptors["unbounded_events"]:
        issues.append("unbounded_tool_output_in_event")
    if output_descriptors["unreferenced_bash_results"]:
        issues.append("bash_result_missing_offload_reference")
    if output_descriptors["oversized_previews"]:
        issues.append("tool_preview_exceeded_limit")
    if watcher.snapshot.namespace != "cache_live_root":
        issues.append("unexpected_watcher_namespace")
    return {
        "event_count": len(events),
        "terminal_run_ids": sorted(expected_run_ids),
        "observed_terminal_run_ids": sorted(terminal_run_ids),
        "terminal_events": [
            {
                "run_id": event.run_id,
                "source_path": list(event.source_path),
                "type": event.type,
                "error_cause": _safe_error_cause(event.data.get("error")),
            }
            for event in terminal_events
        ],
        "tool_events": {
            f"{event_type}:{tool_name}": sum(
                count
                for (
                    _run_id,
                    observed_type,
                    observed_name,
                ), count in tool_events.items()
                if observed_type == event_type and observed_name == tool_name
            )
            for event_type in ("tool.start", "tool.end")
            for tool_name in ("read", "bash", "agent")
        },
        "tool_output_events": output_descriptors,
        "issues": issues,
    }


def _safe_error_cause(error):
    if error is None:
        return None
    message = str(error).lower()
    known_types = (
        "timeout",
        "cancelled",
        "canceled",
        "sqlite",
        "permission",
        "connection",
        "provider",
        "tool",
        "valueerror",
        "typeerror",
        "runtimeerror",
        "keyerror",
        "assertionerror",
        "service recovery",
    )
    for error_type in known_types:
        if error_type in message:
            return error_type.replace(" ", "_")
    return "other_error"


def _summarize_tool_output_events(events, preview_bytes=256):
    unbounded = []
    unreferenced_bash = []
    oversized_previews = []
    for event in events:
        if event.type != "tool.end":
            continue
        data = event.data
        if len(str(data).encode("utf-8")) > 8192:
            unbounded.append(data.get("tool_name", "unknown"))
        if data.get("tool_name") == "bash" and not _contains_output_reference(
            data.get("result")
        ):
            unreferenced_bash.append(event.run_id)
        if _preview_payload_bytes(data.get("result")) > preview_bytes:
            oversized_previews.append(data.get("tool_name", "unknown"))
    return {
        "unbounded_events": unbounded,
        "unreferenced_bash_results": unreferenced_bash,
        "oversized_previews": oversized_previews,
    }


def _contains_output_reference(value):
    if isinstance(value, dict):
        if value.get("type") == "tool_result_reference" or value.get(
            "output_reference"
        ):
            return True
        return any(_contains_output_reference(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_output_reference(item) for item in value)
    return getattr(value, "output_reference", None) is not None


def _preview_payload_bytes(value):
    if isinstance(value, dict):
        own_preview = sum(
            len(value.get(key, "").encode("utf-8"))
            for key in ("preview", "stdout", "stderr")
            if isinstance(value.get(key, ""), str)
        )
        return own_preview + sum(
            _preview_payload_bytes(item) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return sum(_preview_payload_bytes(item) for item in value)
    if hasattr(value, "stdout") or hasattr(value, "stderr"):
        return sum(
            len(getattr(value, key, "").encode("utf-8"))
            for key in ("stdout", "stderr")
            if isinstance(getattr(value, key, ""), str)
        )
    return 0


def _count_child_tool_events(events, child_run_ids):
    tool_events = Counter(
        (event.run_id, event.type, event.data.get("tool_name"))
        for event in events
        if event.type in {"tool.start", "tool.end"}
    )
    issues = [
        f"missing_{tool_name}_{event_type}"
        for run_id in child_run_ids
        for tool_name in ("read", "bash")
        for event_type in ("tool.start", "tool.end")
        if tool_events[run_id, event_type, tool_name] < 1
    ]
    return tool_events, issues


@pytest.mark.asyncio
async def test_drain_until_runs_waits_for_background_terminals_after_root_end():
    class FakeWatcher:
        def __init__(self, events):
            self.events = events

        def __aiter__(self):
            async def iterate():
                for event in self.events:
                    yield event

            return iterate()

    def event(run_id, event_type, source_path=(), data=None):
        return SimpleNamespace(
            run_id=run_id,
            type=event_type,
            source_path=source_path,
            data=data or {},
        )

    watcher = FakeWatcher(
        [
            event("root", "run.end", ("root",)),
            event("shared", "run.end", ("root",)),
            event(
                "shared",
                "run.error",
                ("root", "agent", "io_a"),
                {"error": "RuntimeError: confidential prompt text"},
            ),
            event("child-b", "run.end", ("root", "agent", "io_b")),
            event("later-run", "run.start", ("root", "later")),
        ]
    )

    drained, terminal_events = await _drain_until_runs(
        watcher, {"root", "shared", "child-b"}, quiet_period=0.001
    )

    assert [(item.run_id, item.source_path, item.type) for item in terminal_events] == [
        ("root", ("root",), "run.end"),
        ("shared", ("root",), "run.end"),
        ("shared", ("root", "agent", "io_a"), "run.error"),
        ("child-b", ("root", "agent", "io_b"), "run.end"),
    ]
    assert [(item.run_id, item.type) for item in drained] == [
        ("root", "run.end"),
        ("shared", "run.end"),
        ("shared", "run.error"),
        ("child-b", "run.end"),
        ("later-run", "run.start"),
    ]
    assert _safe_error_cause("RuntimeError: confidential prompt text") == "runtimeerror"


def test_bounded_shell_preview_is_allowed_in_tool_event():
    marker = _NOISE_MARKER
    event = SimpleNamespace(
        type="tool.end",
        run_id="child",
        data={
            "tool_name": "bash",
            "result": {
                "results": [{"stdout": marker, "stderr": ""}],
                "output_reference": {"result_id": "content-id"},
            },
        },
    )

    summary = _summarize_tool_output_events([event], len(marker.encode()))

    assert summary == {
        "unbounded_events": [],
        "unreferenced_bash_results": [],
        "oversized_previews": [],
    }


def _write_metrics(summary, samples, record_property):
    record_property("live_service_cache_policy", json.dumps(summary, sort_keys=True))
    artifact = Path(
        os.getenv(
            "MSGFLUX_LIVE_SERVICE_CACHE_POLICY_ARTIFACT",
            str(Path(tempfile.gettempdir()) / "msgflux-live-service-cache-policy.json"),
        )
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n"
    )


def _peak_metrics(samples):
    keys = (
        "server_rss_bytes",
        "child_rss_bytes",
        "aggregate_rss_bytes",
        "server_fd_count",
        "child_fd_count",
        "child_process_count",
        "loaded_sessions",
        "owned_bundles_open",
        "sqlite_store_bytes",
        "sqlite_wal_bytes",
        "sqlite_pending_tasks",
        "sqlite_running_tasks",
    )
    return {key: max((sample[key] for sample in samples), default=0) for key in keys}


@pytest.mark.skipif(
    os.getenv(_OPT_IN) != "1",
    reason=f"Set {_OPT_IN}=1 to run the bounded live Codex cache stress test",
)
@pytest.mark.asyncio
async def test_live_codex_cache_eviction_keeps_background_sqlite_and_watchers_safe(
    tmp_path, record_property
):
    psutil = pytest.importorskip("psutil")
    mf.load_dotenv(os.getenv("MSGFLUX_TEST_DOTENV", ".env"))

    model = mf.Model.chat_completion(
        "openai-codex/gpt-6-luna",
        api_mode="responses",
        reasoning_effort="medium",
        chat_transport=CodexChatTransport(timeout=90, max_retries=0),
        retry=False,
    )
    managed_root = tmp_path / "managed-agent"
    service_journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(
        store=service_journal,
        cache_policy=SessionCachePolicy(max_loaded=3, idle_timeout=0),
    )
    workspace_roots = {}
    workspaces = {}
    agents_by_thread = {}
    tools_config = ToolOutputOffloadConfig(
        max_inline_bytes=4096,
        preview_bytes=256,
        max_capture_bytes=512_000,
        max_result_bytes=2_000_000,
        max_store_bytes=16_000_000,
    )

    root_prompt = (
        "You are a controller. For every user request, call the `agent` tool "
        "twice: once with name `io_a` and once with name `io_b`. Set "
        "run_in_background=true on both calls. Send both child calls immediately. "
        "Do not wait for either child or ask about their progress. After both "
        "background calls have been accepted, return a short confirmation that "
        "names both children."
    )

    def factory(thread):
        thread_id = thread.thread_id
        workspace_root = workspace_roots[thread_id]
        workspace = workspaces[thread_id]
        children = [
            Agent(
                name=child_name,
                model=model,
                workspace=workspace,
                tools=[ReadFileTool(max_text_bytes=4096), BashTool()],
                extensions=[ToolTurnLimitExtension(6, warn_remaining=0)],
                system_prompt=_child_system_prompt(),
                config={"stream": True},
            )
            for child_name in ("io_a", "io_b")
        ]
        root = Agent(
            name="cache_live_root",
            model=model,
            workspace=workspace,
            agent_dir=managed_root,
            extensions=[
                ManagedToolOutputOffloadExtension(tools_config),
                ToolTurnLimitExtension(6, warn_remaining=0),
            ],
            system_prompt=root_prompt,
            config={"stream": True},
        )
        root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
        for child in children:
            root.tool_library.add(child)
        agents_by_thread[thread_id] = [root, *children]
        return AgentSession(root)

    service.register("cache_live_root", factory)
    threads = {}
    for thread_id in _THREAD_IDS:
        root = tmp_path / thread_id
        root.mkdir()
        payload_path = root / "payload.bin"
        _write_payload(payload_path)
        workspace_roots[thread_id] = root
        workspaces[thread_id] = mf.AgentWorkspace.local(root)
        workspaces[thread_id].aclose = AsyncMock(wraps=workspaces[thread_id].aclose)
        threads[thread_id] = await service.open_thread(
            "cache_live_root", thread_id=thread_id
        )

    samples = []
    sampler_stop = asyncio.Event()
    task_database_paths = [
        managed_root / "threads" / thread_id / "tasks.sqlite3"
        for thread_id in _THREAD_IDS
    ]
    sampler = asyncio.create_task(
        _sample_processes(psutil, service, task_database_paths, sampler_stop, samples)
    )
    watch_stack = AsyncExitStack()
    held_leases = []
    watchers = {}
    admissions = {}
    summary = {
        "model": "openai-codex/gpt-6-luna",
        "reasoning_effort": "medium",
        "root_threads": len(_THREAD_IDS),
        "background_children": 6,
    }
    try:
        async with asyncio.timeout(300):
            # Hold one lease per loaded generation until all three root workers
            # are admitted, then let the zero-timeout cleaner observe them.
            for thread_id in _THREAD_IDS:
                held_leases.append(await service.acquire_session(thread_id))
            model.aclose = AsyncMock(wraps=model.aclose)
            for thread_id in _THREAD_IDS:
                watchers[thread_id] = await watch_stack.enter_async_context(
                    service.watch(thread_id)
                )
            admitted = await asyncio.gather(
                *(
                    service.prompt(
                        thread_id,
                        "Launch both I/O children now and return without waiting.",
                        request_id=f"live-{thread_id}",
                    )
                    for thread_id in _THREAD_IDS
                )
            )
            admissions = dict(zip(_THREAD_IDS, admitted))
            await asyncio.gather(*(lease.aclose() for lease in held_leases))
            held_leases.clear()

            (
                inspections,
                root_receipts,
                peak_running_tasks,
                peak_pending_tasks,
                pending_summary,
            ) = await _wait_for_live_tasks(service, _THREAD_IDS, admissions)
            assert all(receipt.status == "completed" for receipt in root_receipts)
            summary.update(
                {
                    "peak_running_children": peak_running_tasks,
                    "peak_pending_children": peak_pending_tasks,
                    "pending_task_snapshot": pending_summary,
                    "peak_resources": _peak_metrics(samples),
                    "sample_count": len(samples),
                }
            )
            assert peak_running_tasks >= 2, {
                "peak_running_tasks": peak_running_tasks,
                "peak_pending_tasks": peak_pending_tasks,
                "pending_tasks": pending_summary,
            }
            _validate_inspections(inspections, admissions)

            await _wait_for_cache_empty(service, timeout=15)
            integrity, database_metadata, offloaded_counts = _validate_closed_stores(
                managed_root, admissions, inspections, tools_config, workspace_roots
            )

            # Automatic disposal closes the owned SQLite handles before the
            # independent watchers are drained from their retained event buffers.
            resources = [
                agent._owned_threads[thread_id].resources
                for thread_id in _THREAD_IDS
                for agent in agents_by_thread[thread_id][:1]
            ]
            assert len(resources) == 3 and all(item._closed for item in resources)
            assert len(service._cache.sessions) == 0
            assert model.aclose.await_count == 0
            assert all(
                workspace.aclose.await_count == 0 for workspace in workspaces.values()
            )
            assert any(sample["sqlite_read_queries"] for sample in samples)
            assert not any(sample["sqlite_busy_errors"] for sample in samples)
            assert any(sample["sqlite_state_changes_observed"] for sample in samples)
            summary.update(
                {
                    "sqlite_read_queries": sum(
                        sample["sqlite_read_queries"] for sample in samples
                    ),
                    "sqlite_busy_errors": sum(
                        sample["sqlite_busy_errors"] for sample in samples
                    ),
                    "sqlite_state_changes_observed": sum(
                        sample["sqlite_state_changes_observed"] for sample in samples
                    ),
                    "sqlite_state_change_metric": (
                        "task-row changes between 50ms samples; not a write count"
                    ),
                }
            )

            all_events = await _drain_watchers(watchers, admissions, inspections)
            summary["watchers"] = all_events
            assert all(
                not watcher_summary["issues"] for watcher_summary in all_events.values()
            ), all_events

            assert all(workspace is not None for workspace in workspaces.values())

            sampler_stop.set()
            await asyncio.gather(sampler, return_exceptions=True)
            summary = {
                "outcome": "passed",
                "model": "openai-codex/gpt-6-luna",
                "reasoning_effort": "medium",
                "root_threads": len(_THREAD_IDS),
                "background_children": 6,
                "peak_running_children": peak_running_tasks,
                "peak_pending_children": peak_pending_tasks,
                "peak_resources": _peak_metrics(samples),
                "sample_count": len(samples),
                "sqlite_read_queries": sum(
                    sample["sqlite_read_queries"] for sample in samples
                ),
                "sqlite_busy_errors": sum(
                    sample["sqlite_busy_errors"] for sample in samples
                ),
                "sqlite_state_changes_observed": sum(
                    sample["sqlite_state_changes_observed"] for sample in samples
                ),
                "task_summaries": database_metadata,
                "sqlite_integrity": integrity,
                "large_offloaded_artifacts": offloaded_counts,
                "watchers": all_events,
            }
    finally:
        sampler_stop.set()
        await asyncio.gather(sampler, return_exceptions=True)
        failure = sys.exc_info()[1]
        summary.setdefault("outcome", "failed" if failure else "not_completed")
        summary.setdefault("failure_type", type(failure).__name__ if failure else None)
        summary.setdefault("peak_resources", _peak_metrics(samples))
        summary.setdefault("sample_count", len(samples))
        _write_metrics(summary, samples, record_property)
        await asyncio.gather(
            *(lease.aclose() for lease in held_leases), return_exceptions=True
        )
        await watch_stack.aclose()
        await service.aclose()
        service_journal.close()
        for agents in agents_by_thread.values():
            for agent in agents:
                await agent.aclose()
        for workspace in workspaces.values():
            await workspace.aclose()
        await model.aclose()


async def _wait_for_cache_empty(service, *, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while service._cache.sessions:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"Managed cache still holds {len(service._cache.sessions)} sessions"
            )
        await asyncio.sleep(0.05)
