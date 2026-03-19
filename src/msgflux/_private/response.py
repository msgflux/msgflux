import asyncio
import threading
from collections import deque
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


class BaseStreamResponse(CoreResponse):
    def __init__(self, mode: Literal["sync", "async"] = "sync"):
        if mode not in {"sync", "async"}:
            raise ValueError("`mode` must be `sync` or `async`")
        self.mode = mode
        if mode == "async":
            self.first_chunk_event = asyncio.Event()
        else:
            self.first_chunk_event = threading.Event()
        self.data = None
        self._queue = None
        self._queue_loop = None
        self._pending_chunks = deque()
        self._queue_lock = threading.Lock()
        self.metadata = None
        self.response_type = None

    def add(self, data: Any):
        """Add data to the stream queue in a thread-safe way."""
        with self._queue_lock:
            queue = self._queue
            loop = self._queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._pending_chunks.append(data)
                return

        loop.call_soon_threadsafe(queue.put_nowait, data)

    def _bind_consumer_queue(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        with self._queue_lock:
            if self._queue is None:
                self._queue = asyncio.Queue()
                self._queue_loop = loop
                while self._pending_chunks:
                    self._queue.put_nowait(self._pending_chunks.popleft())
            elif self._queue_loop is not loop:
                raise RuntimeError(
                    "BaseStreamResponse.consume() must run on the same event loop."
                )
            return self._queue

    async def consume(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields chunks from the queue until None is received."""
        queue = self._bind_consumer_queue()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
