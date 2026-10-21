"""Read-only CLI projections of registered agents, thread bindings and workers."""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

from msgflux.utils.msgspec import msgspec_dumps


def _write_listing(name, rows, columns, *, as_json):
    if as_json:
        sys.stdout.write(msgspec_dumps({name: rows}) + "\n")
        return
    if not rows:
        sys.stdout.write(f"No {name}.\n")
        return
    cells = [[label for _key, label in columns]] + [
        [str(row[key]) if row[key] is not None else "-" for key, _label in columns]
        for row in rows
    ]
    widths = [max(map(len, column)) for column in zip(*cells, strict=True)]
    for row in cells:
        sys.stdout.write(
            "  ".join(
                value.ljust(width) for value, width in zip(row, widths, strict=True)
            ).rstrip()
            + "\n"
        )


async def _activity(client):
    # Capture workers before durable bindings so newly observed workers' threads
    # are available in the mapping. Neither call constructs Agent sessions.
    runs = await client.active_runs()
    threads = await client.threads()
    return runs, threads


async def list_agents(client, args):
    agents, (runs, threads) = await asyncio.gather(client.agents(), _activity(client))
    bindings = {thread.thread_id: thread.agent_id for thread in threads}
    counts = Counter(
        bindings[run.thread_id] for run in runs if run.thread_id in bindings
    )
    rows = [
        {
            "agent_id": agent,
            "active_runs": counts[agent],
            "state": "running" if counts[agent] else "idle",
        }
        for agent in agents
        if not args.active or counts[agent]
    ]
    _write_listing(
        "agents",
        rows,
        (
            ("agent_id", "AGENT"),
            ("active_runs", "ACTIVE_RUNS"),
            ("state", "STATE"),
        ),
        as_json=args.json,
    )


async def list_threads(client, args):
    runs, threads = await _activity(client)
    counts = Counter(run.thread_id for run in runs)
    rows = [
        {
            "thread_id": thread.thread_id,
            "agent_id": thread.agent_id,
            "cwd": thread.cwd,
            "interactive": thread.interactive,
            "active_runs": counts[thread.thread_id],
            "state": "running" if counts[thread.thread_id] else "idle",
        }
        for thread in threads
        if not args.active or counts[thread.thread_id]
    ]
    _write_listing(
        "threads",
        rows,
        (
            ("thread_id", "THREAD_ID"),
            ("agent_id", "AGENT"),
            ("active_runs", "ACTIVE_RUNS"),
            ("state", "STATE"),
            ("cwd", "CWD"),
        ),
        as_json=args.json,
    )


async def list_active_runs(client, args):
    runs, threads = await _activity(client)
    bindings = {thread.thread_id: thread.agent_id for thread in threads}
    if args.thread_id is not None:
        if args.thread_id not in bindings:
            raise KeyError(f"Unknown thread: {args.thread_id}")
        runs = tuple(run for run in runs if run.thread_id == args.thread_id)
    if args.agent is not None:
        if args.agent not in await client.agents():
            raise KeyError(f"Unknown agent: {args.agent}")
        runs = tuple(run for run in runs if bindings.get(run.thread_id) == args.agent)
    if args.json:
        sys.stdout.write(msgspec_dumps({"runs": runs}) + "\n")
    elif runs:
        rows = [
            {
                "thread_id": run.thread_id,
                "agent_id": bindings.get(run.thread_id),
                "run_id": run.run_id,
                "request_id": run.request_id,
                "status": run.status,
            }
            for run in runs
        ]
        _write_listing(
            "runs",
            rows,
            (
                ("thread_id", "THREAD_ID"),
                ("agent_id", "AGENT"),
                ("run_id", "RUN_ID"),
                ("request_id", "REQUEST_ID"),
                ("status", "STATUS"),
            ),
            as_json=False,
        )
    else:
        sys.stdout.write("No active runs.\n")


async def list_resources(client, args):
    await {"agents": list_agents, "threads": list_threads, "runs": list_active_runs}[
        args.resource
    ](client, args)
