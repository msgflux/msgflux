"""Read final answers without submitting, resuming, or interrupting work."""

import asyncio
import sys

import msgspec

from msgflux.runtime.service import RunResult, ServiceRecoveryRequiredError
from msgflux.runtime.service.http import AgentServiceClient
from msgflux.utils.msgspec import msgspec_dumps


async def _wait_for_result(client: AgentServiceClient, result: RunResult) -> RunResult:
    inspection = await client.inspect_run(result.thread_id, result.run_id)
    if inspection.receipt is None:
        raise ServiceRecoveryRequiredError("Run has no admission receipt to wait for")
    while True:
        receipt = await client.receipt(
            result.thread_id,
            inspection.receipt.request_id,
            require_waitable=True,
        )
        if receipt.status not in {"accepted", "running"}:
            return await client.run_result(result.thread_id, result.run_id)
        await asyncio.sleep(0.05)


async def show_result(client: AgentServiceClient, args) -> None:
    """Resolve a thread once, optionally wait, and print its selected run result."""
    result = (
        await client.thread_result(args.thread_id)
        if args.resource == "threads"
        else await client.run_result(args.thread_id, args.run_id)
    )
    if args.wait and result.status in {"accepted", "running"}:
        try:
            result = await _wait_for_result(client, result)
        except ServiceRecoveryRequiredError as error:
            result = await client.run_result(result.thread_id, result.run_id)
            if result.status != "completed":
                result = msgspec.structs.replace(
                    result, error=result.error or str(error)
                )
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif result.status == "completed":
        sys.stdout.write((result.output or "") + "\n")
    else:
        sys.stderr.write(f"Run {result.run_id}: {result.status}\n")
        if result.error:
            sys.stderr.write(f"{result.error}\n")
    if result.status != "completed":
        raise SystemExit(1)
