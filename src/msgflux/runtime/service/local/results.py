"""Read final answers without submitting, resuming, or interrupting work."""

import asyncio
import sys
from contextlib import nullcontext

import msgspec

from msgflux.runtime.service import RunResult, ServiceRecoveryRequiredError
from msgflux.runtime.service.http import AgentServiceClient
from msgflux.utils.msgspec import msgspec_dumps


def _timed_out(
    result: RunResult, timeout: float, *, settled: str | None = None
) -> RunResult:
    """Describe an expired observation without asking the service again.

    ``result`` is the last state observed without a final answer, so an expired
    budget can never be reported as a completed run.
    """
    observed = (
        f"Observation of run {result.run_id} expired after {timeout:g}s "
        f"while it was {result.status}."
    )
    if settled is not None:
        observed += (
            f" Its receipt already reported {settled}, but the recorded answer "
            "was not read before the budget expired."
        )
    guidance = (
        "The agent was not interrupted. Retrieve the outcome again with "
        f"`msgflux runs result {result.thread_id} {result.run_id}` "
        f"or inspect it with `msgflux runs inspect {result.thread_id} "
        f"{result.run_id}`."
    )
    return msgspec.structs.replace(result, error=f"{observed} {guidance}")


async def _wait_for_result(
    client: AgentServiceClient,
    result: RunResult,
    *,
    timeout: float | None = None,
) -> RunResult:
    """Wait for one selected run, optionally bounding observation only.

    The deadline covers every lookup performed here, including the recovery
    fallback, and is never sent to the daemon, so an expired budget leaves the
    agent running. Until a final result is read, the diagnostic keeps the last
    unfinished ``RunResult`` observed.
    """
    budget = None if timeout is None else asyncio.get_running_loop().time() + timeout
    guard = nullcontext() if budget is None else asyncio.timeout_at(budget)
    observed, settled = result, None

    async def observe() -> RunResult:
        nonlocal observed, settled
        inspection = await client.inspect_run(result.thread_id, result.run_id)
        if inspection.receipt is None:
            raise ServiceRecoveryRequiredError(
                "Run has no admission receipt to wait for"
            )
        if inspection.receipt.status in {"accepted", "running"}:
            observed = msgspec.structs.replace(
                observed, status=inspection.receipt.status
            )
        while True:
            receipt = await client.receipt(
                result.thread_id,
                inspection.receipt.request_id,
                require_waitable=True,
            )
            if receipt.status not in {"accepted", "running"}:
                settled = receipt.status
                return await client.run_result(result.thread_id, result.run_id)
            observed = msgspec.structs.replace(observed, status=receipt.status)
            await asyncio.sleep(0.05)

    try:
        async with guard:
            try:
                return await observe()
            except ServiceRecoveryRequiredError as error:
                # The recovery lookup stays inside the same observation budget.
                recovered = await client.run_result(result.thread_id, result.run_id)
                if recovered.status == "completed":
                    return recovered
                return msgspec.structs.replace(
                    recovered, error=recovered.error or str(error)
                )
    except TimeoutError:
        # Only this context's own deadline is a diagnostic. A transport timeout
        # raised by the client before it elapses stays an error.
        if (
            timeout is not None
            and isinstance(guard, asyncio.Timeout)
            and guard.expired()
        ):
            return _timed_out(observed, timeout, settled=settled)
        raise


async def show_result(client: AgentServiceClient, args) -> None:
    """Resolve a thread once, optionally wait, and print its selected run result."""
    result = (
        await client.thread_result(args.thread_id)
        if args.resource == "threads"
        else await client.run_result(args.thread_id, args.run_id)
    )
    if args.wait and result.status in {"accepted", "running"}:
        result = await _wait_for_result(
            client, result, timeout=getattr(args, "wait_timeout", None)
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
