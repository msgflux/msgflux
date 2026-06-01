import asyncio
import threading
from collections import deque
from typing import Any, AsyncGenerator, Literal, Optional, Union


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
        self.reasoning = None
        self.metadata = None
        self.response_type = None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None

    def add(self, data: Any):
        self.data = data

    def consume(self) -> Any:
        return self.data

    def consume_reasoning(self) -> Optional[str]:
        return self.reasoning


class BaseStreamResponse(CoreResponse):
    def __init__(self, mode: Literal["sync", "async"] = "sync"):
        if mode not in {"sync", "async"}:
            raise ValueError("`mode` must be `sync` or `async`")
        self.mode = mode
        if mode == "async":
            self.first_chunk_event = asyncio.Event()
            self._response_type_event = asyncio.Event()
        else:
            self.first_chunk_event = threading.Event()
            self._response_type_event = threading.Event()
        self.data = None
        self.reasoning = None
        self.has_reasoning = False

        # Content queue
        self._queue = None
        self._queue_loop = None
        self._pending_chunks = deque()
        self._queue_lock = threading.Lock()

        # Reasoning queue
        self._reasoning_queue = None
        self._reasoning_queue_loop = None
        self._reasoning_pending_chunks = deque()
        self._reasoning_queue_lock = threading.Lock()

        self.metadata = None
        self.response_type = None
        self.error = None

    def _finish_queue_with_none(
        self,
        *,
        queue_attr: str,
        loop_attr: str,
        pending_attr: str,
        lock_attr: str,
    ) -> None:
        with getattr(self, lock_attr):
            queue = getattr(self, queue_attr)
            loop = getattr(self, loop_attr)
            pending = getattr(self, pending_attr)
            if queue is None or loop is None or loop.is_closed():
                pending.append(None)
                return

        loop.call_soon_threadsafe(queue.put_nowait, None)

    def _fail_stream(self, error: Exception) -> None:
        self.set_error(error)
        self._finish_queue_with_none(
            queue_attr="_queue",
            loop_attr="_queue_loop",
            pending_attr="_pending_chunks",
            lock_attr="_queue_lock",
        )
        self._finish_queue_with_none(
            queue_attr="_reasoning_queue",
            loop_attr="_reasoning_queue_loop",
            pending_attr="_reasoning_pending_chunks",
            lock_attr="_reasoning_queue_lock",
        )

    def _accumulate_data(self, data: Any) -> None:
        if data is None:
            return

        if not isinstance(data, (str, bytes)):
            raise TypeError(
                "ModelStreamResponse only supports `str` or `bytes` chunks, "
                f"got `{type(data).__name__}`."
            )

        if self.data is None:
            self.data = data
            return

        if isinstance(self.data, str) and isinstance(data, str):
            self.data += data
            return

        if isinstance(self.data, bytes) and isinstance(data, bytes):
            self.data += data
            return

        raise TypeError(
            "ModelStreamResponse received mixed chunk types: "
            f"`{type(self.data).__name__}` then `{type(data).__name__}`."
        )

    def set_response_type(self, response_type: str):
        super().set_response_type(response_type)
        if not self._response_type_event.is_set():
            self._response_type_event.set()

    def set_error(self, error: Exception):
        self.error = error
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        if not self._response_type_event.is_set():
            self._response_type_event.set()

    def add(self, data: Any):
        """Add data to the content stream queue in a thread-safe way."""
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()

        try:
            self._accumulate_data(data)
        except Exception as e:
            self._fail_stream(e)
            raise

        with self._queue_lock:
            queue = self._queue
            loop = self._queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._pending_chunks.append(data)
                return

        loop.call_soon_threadsafe(queue.put_nowait, data)

    def add_reasoning(self, data: Any):
        """Add data to the reasoning stream queue in a thread-safe way."""
        if data is not None and not self.has_reasoning:
            self.has_reasoning = True
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        with self._reasoning_queue_lock:
            queue = self._reasoning_queue
            loop = self._reasoning_queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._reasoning_pending_chunks.append(data)
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

    def _bind_reasoning_queue(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        with self._reasoning_queue_lock:
            if self._reasoning_queue is None:
                self._reasoning_queue = asyncio.Queue()
                self._reasoning_queue_loop = loop
                while self._reasoning_pending_chunks:
                    self._reasoning_queue.put_nowait(
                        self._reasoning_pending_chunks.popleft()
                    )
            elif self._reasoning_queue_loop is not loop:
                raise RuntimeError(
                    "BaseStreamResponse.consume_reasoning() "
                    "must run on the same event loop."
                )
            return self._reasoning_queue

    async def next_chunk(self) -> Optional[Union[bytes, str]]:
        """Return the next content chunk, or None when the stream is complete."""
        queue = self._bind_consumer_queue()
        chunk = await queue.get()
        if chunk is None:
            if self.error is not None:
                raise self.error
            return None
        return chunk

    async def consume(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields content chunks until None is received."""
        while True:
            chunk = await self.next_chunk()
            if chunk is None:
                break
            yield chunk

    async def consume_reasoning(self) -> AsyncGenerator[str, None]:
        """Async generator that yields reasoning chunks until None is received."""
        queue = self._bind_reasoning_queue()
        while True:
            chunk = await queue.get()
            if chunk is None:
                if self.error is not None:
                    raise self.error
                break
            yield chunk
