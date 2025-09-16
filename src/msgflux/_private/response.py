import asyncio
from typing import Any, AsyncGenerator, Literal, Union


class CoreResponse:
    def set_metadata(self, metadata: Any):
        self.metadata = metadata

    def set_response_type(self, response_type: str):
        if isinstance(response_type, str):
            self.response_type = response_type
        else:
            raise TypeError(
                f"`response_type` requires strgiven `{type(response_type)}`"
            )


class BaseResponse(CoreResponse):

    def __init__(self):
        self.data = None
        self.metadata = None
        self.response_type = None

    def add(self, data: Any):
        self.data = data

    def consume(self) -> Any:
        return self.data


class StreamResponse(CoreResponse):

    def __init__(self):
        self.first_chunk_event = asyncio.Event()
        self.queue = asyncio.Queue()
        self.metadata = None
        self.response_type = None

    def add(self, data: Any):
        """Add data to the stream queue (async)."""
        self.queue.put_nowait(data)

    async def consume(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields chunks from the queue until None is received."""
        while True:
            try:
                chunk = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                continue
