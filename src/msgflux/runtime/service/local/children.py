"""Bounded observation of directly delegated child threads."""

from __future__ import annotations

import asyncio
import sys

import msgspec

from msgflux.runtime.service.local.listing import _write_listing
from msgflux.utils.msgspec import msgspec_dumps

_SETTLED = frozenset({"completed", "failed", "paused", "interrupted"})


def _summary(child):
    receipt = child.receipt
    return {
        "thread_id": child.thread.thread_id,
        "agent_id": child.thread.agent_id,
        "model": child.thread.model,
        "run_id": receipt.run_id if receipt is not None else None,
        "request_id": receipt.request_id if receipt is not None else None,
        "status": receipt.status if receipt is not None else "unstarted",
    }


async def _wait_for_child(client, args, first):
    excluded = set(args.exclude_run or ())
    targets = {
        child.thread.thread_id: child
        for child in first.children
        if child.receipt is not None and child.receipt.run_id not in excluded
    }
    if not targets:
        return {
            "parent_thread_id": args.thread_id,
            "status": "empty",
            "next_cursor": first.next_cursor,
        }
    current = first
    while True:
        by_id = {child.thread.thread_id: child for child in current.children}
        for thread_id, original in targets.items():
            child = by_id.get(thread_id)
            receipt = child.receipt if child is not None else None
            if receipt is None or receipt.run_id != original.receipt.run_id:
                # Pin the initial request when a child has already moved on.
                receipt = await client.receipt(thread_id, original.receipt.request_id)
                child = msgspec.structs.replace(original, receipt=receipt)
            if receipt.status in _SETTLED:
                return {
                    "parent_thread_id": args.thread_id,
                    "status": "settled",
                    "child": _summary(child),
                    "next_cursor": first.next_cursor,
                }
        await asyncio.sleep(0.1)
        current = await client.child_threads(
            args.thread_id, limit=args.limit or 100, after=args.after_child or 0
        )


async def show_children(client, args):
    guard = asyncio.timeout(args.wait_timeout if args.wait else None)
    try:
        async with guard:
            page = await client.child_threads(
                args.thread_id, limit=args.limit or 100, after=args.after_child or 0
            )
            if args.wait:
                result = await _wait_for_child(client, args, page)
            else:
                result = page
    except TimeoutError:
        if not guard.expired():
            raise
        result = {"parent_thread_id": args.thread_id, "status": "timeout"}
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif args.wait:
        child = result.get("child")
        if child is not None:
            sys.stdout.write(
                f"Child {child['thread_id']}, run {child['run_id']}: "
                f"{child['status']}\n"
            )
        else:
            sys.stdout.write(
                f"Selected children of {args.thread_id}: {result['status']}\n"
            )
        if result.get("next_cursor") is not None:
            sys.stdout.write(f"Next cursor: {result['next_cursor']}\n")
    else:
        _write_listing(
            "children",
            [_summary(child) for child in result.children],
            (
                ("thread_id", "THREAD_ID"),
                ("agent_id", "AGENT"),
                ("run_id", "RUN_ID"),
                ("status", "STATUS"),
            ),
            as_json=False,
        )
        if result.next_cursor is not None:
            sys.stdout.write(f"Next cursor: {result.next_cursor}\n")
    if args.wait and result["status"] == "timeout":
        raise SystemExit(1)
