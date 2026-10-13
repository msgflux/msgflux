"""Noninteractive coding requests observed through the service's watch API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

import httpx2
import msgspec

from msgflux.runtime.service.local import connect_local_service

_TERMINAL_EVENTS = {"run.end", "run.error", "run.paused", "run.interrupted"}
_TRANSPORT_ERRORS = (httpx2.TransportError, OSError)

ObservationCallback = Callable[[Literal["snapshot", "event"], Any], None]


class PromptResult(msgspec.Struct):
    """One final CLI JSON record, including identities needed for recovery."""

    status: str = "failed"
    thread_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    run_id: str | None = None
    output: str | None = None
    error: str | None = None
    version: int = 1


class PromptExecution:
    """Own only frontend observation; the daemon owns the admitted execution."""

    def __init__(self) -> None:
        self.result = PromptResult()

    async def run(
        self,
        prompt: str,
        *,
        config: Path | None = None,
        runtime_dir: Path | None = None,
        cwd: Path | None = None,
        agent_id: str | None = None,
        account_name: str | None = None,
        thread_id: str | None = None,
        observer: ObservationCallback | None = None,
        interrupt_on_cancel: bool = False,
    ) -> PromptResult:
        client = await connect_local_service(
            config=config, runtime_dir=runtime_dir, cwd=cwd
        )
        admission = None
        try:
            thread = await client.open_thread(
                agent_id,
                thread_id=thread_id,
                cwd=cwd if thread_id else cwd or Path.cwd(),
                interactive=False,
            )
            self.result.thread_id = thread.thread_id
            self.result.agent_id = thread.agent_id
            self.result.request_id = uuid4().hex
            try:
                # Attach before admitting so short runs cannot outrun observation.
                async with client.watch(thread.thread_id) as watcher:
                    if observer is not None:
                        observer("snapshot", watcher.snapshot)
                    admission = asyncio.create_task(
                        client.prompt(
                            thread.thread_id,
                            prompt,
                            request_id=self.result.request_id,
                            account_name=account_name,
                        )
                    )
                    receipt = await asyncio.shield(admission)
                    self.result.run_id = receipt.run_id
                    await self._observe(client, watcher, observer)
            except _TRANSPORT_ERRORS:
                if self.result.run_id is None:
                    raise
            return await self._wait_result(client, thread.thread_id, observer)
        except asyncio.CancelledError:
            await self._on_cancel(client, admission, interrupt=interrupt_on_cancel)
            raise
        finally:
            if admission is not None:
                if not admission.done():
                    admission.cancel()
                await asyncio.gather(admission, return_exceptions=True)
            await client.aclose()

    async def _wait_result(
        self, client, thread_id: str, observer: ObservationCallback | None
    ) -> PromptResult:
        failures = 0
        while True:
            try:
                receipt = await client.receipt(
                    thread_id, self.result.request_id, require_waitable=True
                )
                if receipt.status not in {"accepted", "running"}:
                    settled = await client.run_result(thread_id, receipt.run_id)
                    self.result.status = settled.status
                    self.result.output = settled.output
                    self.result.error = settled.error
                    return self.result
                async with client.watch(thread_id) as watcher:
                    if observer is not None:
                        observer("snapshot", watcher.snapshot)
                    await self._observe(client, watcher, observer)
                await asyncio.sleep(0.05)
            except _TRANSPORT_ERRORS:
                failures += 1
                if failures >= 3:
                    raise
                await asyncio.sleep(0.2 * failures)

    async def _on_cancel(self, client, admission, *, interrupt: bool) -> None:
        self.result.status = "detached"
        self.result.error = (
            "Observation interrupted; an admitted run remains owned by the service."
        )
        try:
            async with asyncio.timeout(5):
                # The service may have admitted a prompt before its HTTP response
                # reaches us. Recover that same receipt, never resubmit.
                if admission is not None and self.result.run_id is None:
                    receipt = await asyncio.shield(admission)
                    self.result.run_id = receipt.run_id
                if not interrupt:
                    return
                if self.result.run_id is None:
                    self.result.status = "interrupted"
                    self.result.error = "Interrupted before a run was admitted."
                    return
                requested = await client.interrupt(
                    self.result.thread_id, self.result.run_id
                )
                self.result.status = "interrupt_requested" if requested else "detached"
                self.result.error = (
                    "Interruption requested; the run has not settled yet."
                    if requested
                    else "The service did not signal an active run; "
                    "interruption is not confirmed."
                )
                while True:
                    receipt = await client.receipt(
                        self.result.thread_id,
                        self.result.request_id,
                        require_waitable=True,
                    )
                    if receipt.status not in {"accepted", "running"}:
                        settled = await client.run_result(
                            self.result.thread_id, self.result.run_id
                        )
                        self.result.status = settled.status
                        self.result.output = settled.output
                        self.result.error = settled.error
                        return
                    await asyncio.sleep(0.05)
        except TimeoutError:
            # Keep the truthful last state: detached or interrupt_requested.
            return
        except Exception as error:
            self.result.error = f"Could not confirm interruption: {error}"

    async def _observe(
        self, client, watcher, observer: ObservationCallback | None = None
    ) -> None:
        async def receipt_status():
            while True:
                receipt = await client.receipt(
                    self.result.thread_id, self.result.request_id, require_waitable=True
                )
                if receipt.status not in {"accepted", "running"}:
                    return
                await asyncio.sleep(0.25)

        # Receipt settlement can occur after run.end or before a reconnect's
        # snapshot. Never wait indefinitely for an event that has already passed.
        tasks = {
            asyncio.create_task(receipt_status()),
            asyncio.create_task(self._consume_events(watcher, observer)),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _consume_events(
        self, watcher, observer: ObservationCallback | None
    ) -> None:
        async for event in watcher:
            if observer is not None:
                observer("event", event)
            if (
                event.run_id == self.result.run_id
                and len(event.source_path) <= 1
                and event.type in _TERMINAL_EVENTS
            ):
                return
