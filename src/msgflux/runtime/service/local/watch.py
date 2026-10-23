"""Observe an existing thread without submitting or interrupting work."""

from msgflux._private.cli_output import write_observation
from msgflux.runtime.service.http import AgentServiceClient


async def watch_thread(
    client: AgentServiceClient, thread_id: str, *, as_json: bool
) -> None:
    async with client.watch(thread_id) as watcher:
        write_observation("snapshot", watcher.snapshot, as_json=as_json)
        async for event in watcher:
            write_observation("event", event, as_json=as_json)
