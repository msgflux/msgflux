import asyncio
import inspect
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Optional, Union

from msgflux.chat_stream_accumulator import ChatStreamAccumulator


@dataclass(frozen=True)
class StreamFinalState:
    status: Literal["completed", "failed", "interrupted"]
    response_type: str | None
    output: Any
    reasoning: str | None
    reasoning_summary: str | None
    metadata: Any
    error: Exception | None
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class LMStreamEvent:
    """One provider-ordered event produced by a streaming language model."""

    type: Literal[
        "output.delta",
        "reasoning.delta",
        "reasoning_summary.delta",
    ]
    data: Any


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

    def get_tool_intents(self):
        """Return provider-neutral tool intents decoded by the Model."""
        get_intents = getattr(self.data, "get_intents", None)
        if not callable(get_intents):
            return ()
        return get_intents()

    def render_tool_outcomes(self, outcomes):
        """Render runtime outcomes for this response's provider protocol."""
        render = getattr(self.data, "render_outcomes", None)
        if not callable(render):
            raise TypeError("Model response cannot render tool outcomes")
        return render(outcomes)


class BaseResponse(CoreResponse):
    def __init__(self):
        self.data = None
        self.reasoning = None
        self.reasoning_summary = None
        self.history_items = []
        self.metadata = None
        self.response_type = None

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None

    @property
    def has_reasoning_summary(self) -> bool:
        return self.reasoning_summary is not None

    def add(self, data: Any):
        self.data = data

    def consume(self) -> Any:
        return self.data

    def consume_reasoning(self) -> Optional[str]:
        return self.reasoning

    def consume_reasoning_summary(self) -> Optional[str]:
        """Return a provider-generated reasoning summary, when available."""
        return self.reasoning_summary


class BaseStreamResponse(CoreResponse):
    def __init__(self, mode: Literal["sync", "async"] = "sync"):
        if mode not in {"sync", "async"}:
            raise ValueError("`mode` must be `sync` or `async`")
        self.mode = mode
        if mode == "async":
            self.first_chunk_event = asyncio.Event()
            self._response_type_event = asyncio.Event()
            self.reasoning_summary_event = asyncio.Event()
        else:
            self.first_chunk_event = threading.Event()
            self._response_type_event = threading.Event()
            self.reasoning_summary_event = threading.Event()
        self.data = None
        self.reasoning = None
        self.reasoning_summary = None
        self.has_reasoning = False
        self.has_reasoning_summary = False

        # Content queue
        self._queue = None
        self._queue_loop = None
        self._pending_chunks = deque()
        self._queue_lock = threading.Lock()
        self._content_closed = False

        # Reasoning queue
        self._reasoning_queue = None
        self._reasoning_queue_loop = None
        self._reasoning_pending_chunks = deque()
        self._reasoning_queue_lock = threading.Lock()
        self._reasoning_closed = False

        # Reasoning-summary queue
        self._reasoning_summary_queue = None
        self._reasoning_summary_queue_loop = None
        self._reasoning_summary_pending_chunks = deque()
        self._reasoning_summary_queue_lock = threading.Lock()
        self._reasoning_summary_closed = False

        # Provider-ordered event journal. The response owns this bounded-lifetime
        # replay so independent runtime and direct consumers preserve relative
        # order without competing for one queue.
        self._events = []
        self._event_subscribers = []
        self._event_queue_lock = threading.Lock()
        self._events_closed = False

        self.metadata = None
        self.response_type = None
        self.error = None
        self._finalizers = []
        self._pending_finalizer_awaitables = []
        self._consumer_finalizers = []
        self._finalized = False
        self._consumer_finalized = False
        self._final_status = None
        self._finalizer_lock = threading.Lock()
        self.chat_accumulator = ChatStreamAccumulator()

    def _finish_queue_with_none(
        self,
        *,
        queue_attr: str,
        loop_attr: str,
        pending_attr: str,
        lock_attr: str,
        closed_attr: str,
    ) -> None:
        with getattr(self, lock_attr):
            if getattr(self, closed_attr):
                return
            setattr(self, closed_attr, True)
            queue = getattr(self, queue_attr)
            loop = getattr(self, loop_attr)
            pending = getattr(self, pending_attr)
            if queue is None or loop is None or loop.is_closed():
                pending.append(None)
                return

        loop.call_soon_threadsafe(queue.put_nowait, None)

    def _fail_stream(self, error: Exception) -> None:
        self.set_error(error)
        self._close_stream_queues()

    def _close_stream_queues(self) -> None:
        self.finish_reasoning()
        self.finish_reasoning_summary()
        self._finish_content()
        self._finish_events()

    def _finish_content(self) -> None:
        self._finish_queue_with_none(
            queue_attr="_queue",
            loop_attr="_queue_loop",
            pending_attr="_pending_chunks",
            lock_attr="_queue_lock",
            closed_attr="_content_closed",
        )

    def _finish_events(self) -> None:
        with self._event_queue_lock:
            if self._events_closed:
                return
            self._events_closed = True
            subscribers = tuple(self._event_subscribers)
            self._event_subscribers.clear()
        for loop, queue in subscribers:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, None)

    def _add_event(self, event: LMStreamEvent) -> None:
        with self._event_queue_lock:
            if self._events_closed:
                raise RuntimeError("Cannot add an event to a closed stream.")
            self._events.append(event)
            subscribers = tuple(self._event_subscribers)
        for loop, queue in subscribers:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, event)

    def finish_reasoning(self) -> None:
        """Close the reasoning stream without finalizing the content stream."""
        self._finish_queue_with_none(
            queue_attr="_reasoning_queue",
            loop_attr="_reasoning_queue_loop",
            pending_attr="_reasoning_pending_chunks",
            lock_attr="_reasoning_queue_lock",
            closed_attr="_reasoning_closed",
        )

    def finish_reasoning_summary(self) -> None:
        """Close the reasoning-summary stream independently from content."""
        self._finish_queue_with_none(
            queue_attr="_reasoning_summary_queue",
            loop_attr="_reasoning_summary_queue_loop",
            pending_attr="_reasoning_summary_pending_chunks",
            lock_attr="_reasoning_summary_queue_lock",
            closed_attr="_reasoning_summary_closed",
        )
        if not self.reasoning_summary_event.is_set():
            self.reasoning_summary_event.set()

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

    def add_finalizer(self, finalizer) -> None:
        final_state = None
        with self._finalizer_lock:
            if self._finalized:
                final_state = self._build_final_state()
            else:
                self._finalizers.append(finalizer)
        if final_state is not None:
            finalizer(final_state)

    def _add_consumer_finalizer(self, finalizer) -> None:
        """Run a callback after an owner has drained all stream queues."""
        final_state = None
        with self._finalizer_lock:
            if self._consumer_finalized:
                final_state = self._build_final_state()
            else:
                self._consumer_finalizers.append(finalizer)
        if final_state is not None:
            finalizer(final_state)

    def _run_consumer_finalizers(self) -> None:
        with self._finalizer_lock:
            if self._consumer_finalized:
                return
            self._consumer_finalized = True
            finalizers = list(self._consumer_finalizers)
            self._consumer_finalizers.clear()
            final_state = self._build_final_state()

        for finalizer in finalizers:
            finalizer(final_state)

    def _is_finalized(self) -> bool:
        with self._finalizer_lock:
            return self._finalized

    def finish(
        self,
        *,
        error: Exception | None = None,
        status: Literal["completed", "failed", "interrupted"] | None = None,
    ) -> None:
        if self._is_finalized():
            return
        if error is not None:
            self.set_error(error)
        if status is None:
            status = "failed" if self.error is not None else "completed"
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        self._close_stream_queues()
        self._run_finalizers(status=status)

    def _build_final_state(
        self,
        *,
        status: Literal["completed", "failed", "interrupted"] | None = None,
    ) -> StreamFinalState:
        resolved_status = status
        if resolved_status is None:
            resolved_status = self._final_status
        if resolved_status is None:
            resolved_status = "failed" if self.error is not None else "completed"
        return StreamFinalState(
            status=resolved_status,
            response_type=self.response_type,
            output=self.data,
            reasoning=self.reasoning,
            reasoning_summary=self.reasoning_summary,
            metadata=self.metadata,
            error=self.error,
            items=self.chat_accumulator.snapshot(
                fallback_output=self.data,
                fallback_reasoning=self.reasoning,
            ),
        )

    def _run_finalizers(
        self,
        *,
        status: Literal["completed", "failed", "interrupted"],
    ) -> None:
        with self._finalizer_lock:
            if self._finalized:
                return
            self._finalized = True
            self._final_status = status
            finalizers = list(self._finalizers)
            self._finalizers.clear()
            final_state = self._build_final_state(status=status)

        for finalizer in finalizers:
            result = finalizer(final_state)
            if inspect.isawaitable(result):
                self._pending_finalizer_awaitables.append(result)

    async def _await_pending_finalizers(self) -> None:
        """Finish async callbacks emitted by a synchronous stream producer."""
        with self._finalizer_lock:
            pending = list(self._pending_finalizer_awaitables)
            self._pending_finalizer_awaitables.clear()
        if pending:
            await asyncio.gather(*pending)

    def add(self, data: Any, *, accumulate_history: bool = True):
        """Add data to the content stream queue in a thread-safe way."""
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()

        try:
            self._accumulate_data(data)
            if accumulate_history and isinstance(data, str):
                self.chat_accumulator.add_text(data)
        except Exception as e:
            self._fail_stream(e)
            raise

        with self._queue_lock:
            if self._content_closed:
                raise RuntimeError("Cannot add content chunk to a closed stream.")
            queue = self._queue
            loop = self._queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._pending_chunks.append(data)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, data)
        self._add_event(LMStreamEvent(type="output.delta", data=data))

    def add_reasoning(
        self,
        data: Any,
        *,
        history_kind: str = "text",
        item_id: str | None = None,
    ):
        """Add data to the reasoning stream queue in a thread-safe way."""
        if data is not None:
            if history_kind == "summary":
                self.chat_accumulator.add_reasoning(summary=str(data), item_id=item_id)
            else:
                self.chat_accumulator.add_reasoning(str(data), item_id=item_id)
        if data is not None and not self.has_reasoning:
            self.has_reasoning = True
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        with self._reasoning_queue_lock:
            if self._reasoning_closed:
                raise RuntimeError("Cannot add reasoning chunk to a closed stream.")
            queue = self._reasoning_queue
            loop = self._reasoning_queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._reasoning_pending_chunks.append(data)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, data)
        if data is not None:
            self._add_event(LMStreamEvent(type="reasoning.delta", data=data))

    def add_reasoning_summary(self, data: Any, *, item_id: str | None = None):
        """Add a summary delta without presenting it as chain-of-thought."""
        if data is not None:
            self.chat_accumulator.add_reasoning(summary=str(data), item_id=item_id)
            self.has_reasoning_summary = True
            if not self.reasoning_summary_event.is_set():
                self.reasoning_summary_event.set()
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        with self._reasoning_summary_queue_lock:
            if self._reasoning_summary_closed:
                raise RuntimeError(
                    "Cannot add reasoning-summary chunk to a closed stream."
                )
            queue = self._reasoning_summary_queue
            loop = self._reasoning_summary_queue_loop
            if queue is None or loop is None or loop.is_closed():
                self._reasoning_summary_pending_chunks.append(data)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, data)
        if data is not None:
            self._add_event(LMStreamEvent(type="reasoning_summary.delta", data=data))

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

    def _bind_reasoning_summary_queue(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        with self._reasoning_summary_queue_lock:
            if self._reasoning_summary_queue is None:
                self._reasoning_summary_queue = asyncio.Queue()
                self._reasoning_summary_queue_loop = loop
                while self._reasoning_summary_pending_chunks:
                    self._reasoning_summary_queue.put_nowait(
                        self._reasoning_summary_pending_chunks.popleft()
                    )
            elif self._reasoning_summary_queue_loop is not loop:
                raise RuntimeError(
                    "BaseStreamResponse.consume_reasoning_summary() must run on "
                    "the same event loop."
                )
            return self._reasoning_summary_queue

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
        try:
            while True:
                chunk = await self.next_chunk()
                if chunk is None:
                    break
                yield chunk
        finally:
            await self._await_pending_finalizers()

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

    async def consume_reasoning_summary(self) -> AsyncGenerator[str, None]:
        """Yield reasoning-summary chunks without conflating them with CoT."""
        queue = self._bind_reasoning_summary_queue()
        while True:
            chunk = await queue.get()
            if chunk is None:
                if self.error is not None:
                    raise self.error
                break
            yield chunk

    async def consume_events(self) -> AsyncGenerator[LMStreamEvent, None]:
        """Yield language-model events in their original provider order."""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        subscriber = (loop, queue)
        with self._event_queue_lock:
            events = tuple(self._events)
            closed = self._events_closed
            if not closed:
                self._event_subscribers.append(subscriber)
        try:
            for event in events:
                yield event
            if closed:
                if self.error is not None:
                    raise self.error
                return
            while True:
                event = await queue.get()
                if event is None:
                    if self.error is not None:
                        raise self.error
                    return
                yield event
        finally:
            with self._event_queue_lock:
                if subscriber in self._event_subscribers:
                    self._event_subscribers.remove(subscriber)
            await self._await_pending_finalizers()
