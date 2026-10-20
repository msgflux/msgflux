from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Optional, Union

from msgflux.chat_stream_accumulator import ChatStreamAccumulator
from msgflux.exceptions import AbortRequestedError
from msgflux.logger import init_logger

logger = init_logger(__name__)
_STREAM_CONSUMER_OWNERSHIP_LOCK = threading.RLock()


@dataclass(frozen=True)
class StreamFinalState:
    status: Literal["completed", "failed", "interrupted"]
    response_type: str | None
    output: Any
    reasoning: str | None
    reasoning_summary: str | None
    metadata: Any
    error: BaseException | None
    items: list[dict[str, Any]]


@dataclass(frozen=True)
class LMStreamEvent:
    """One provider-ordered event produced by a streaming language model."""

    type: Literal[
        "tool.start",
        "tool.update",
        "tool.end",
        "tools.updated",
        "output.delta",
        "commentary.delta",
        "reasoning.delta",
        "reasoning_summary.delta",
        "message.end",
        "message.start",
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
        self.commentary = []
        self.history_items = []
        self.events: list[LMStreamEvent] = []
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


class _StreamConsumerState:
    """Track active content/event readers across one linked response chain."""

    def __init__(self):
        self.active_consumers = 0
        self.cleanup_started = False
        self.cleanup_complete = False

    def acquire(self) -> None:
        with _STREAM_CONSUMER_OWNERSHIP_LOCK:
            if self.cleanup_started and not self.cleanup_complete:
                raise RuntimeError(
                    "Cannot consume a stream after continuation cleanup started."
                )
            self.active_consumers += 1

    def release(self, *, cleanup: bool) -> bool:
        with _STREAM_CONSUMER_OWNERSHIP_LOCK:
            if self.active_consumers <= 0:
                raise RuntimeError("Stream consumer ownership count underflow.")
            self.active_consumers -= 1
            if cleanup and self.active_consumers == 0:
                self.cleanup_started = True
                return True
        return False

    def mark_cleanup_complete(self) -> None:
        with _STREAM_CONSUMER_OWNERSHIP_LOCK:
            self.cleanup_complete = True


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
        self.commentary = []
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
        self._source_error = None
        self._finalizers = []
        self._pending_finalizer_awaitables = []
        self._consumer_finalizers = []
        self._finalized = False
        self._consumer_finalized = False
        self._final_status = None
        self._continuation: BaseStreamResponse | None = None
        self._is_continuation_member = False
        self._finalizer_lock = threading.Lock()
        self._consumer_ownership = _StreamConsumerState()
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

    def set_error(self, error: BaseException):
        self.error = error
        self._source_error = error
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        if not self._response_type_event.is_set():
            self._response_type_event.set()

    def add_finalizer(self, finalizer) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        context = contextvars.copy_context()
        callback = finalizer

        async def await_result(result):
            return await result

        def finalizer(state):
            result = callback(state)
            if inspect.isawaitable(result):
                if loop is None:
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError(
                        "Async stream finalizers require an event loop at registration"
                    )
                return context.run(
                    asyncio.run_coroutine_threadsafe, await_result(result), loop
                )
            return result

        final_state = None
        with self._finalizer_lock:
            if self._finalized:
                final_state = self._build_final_state()
            else:
                self._finalizers.append(finalizer)
        if final_state is not None:
            result = finalizer(final_state)
            if inspect.isawaitable(result) or isinstance(result, Future):
                with self._finalizer_lock:
                    self._pending_finalizer_awaitables.append(result)

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

    def _set_continuation(self, result: Any) -> None:
        """Link a subsequent stream or terminal result to this response."""
        if isinstance(result, BaseStreamResponse):
            continuation = result
        else:
            continuation = BaseStreamResponse(mode="async")
            continuation._settled_output = result
            if isinstance(result, (str, bytes)):
                continuation.add(result)
            continuation.finish()

        with _STREAM_CONSUMER_OWNERSHIP_LOCK:
            with self._finalizer_lock:
                if self._continuation is not None:
                    raise RuntimeError("A stream continuation is already attached.")
                if continuation is self or continuation._is_continuation_member:
                    raise ValueError(
                        "A stream continuation cannot be reused or cyclic."
                    )

                nodes = []
                pending = [continuation]
                while pending:
                    node = pending.pop()
                    if node is self or node in nodes:
                        raise ValueError(
                            "A stream continuation cannot be reused or cyclic."
                        )
                    nodes.append(node)
                    with node._finalizer_lock:
                        child = node._continuation
                    if child is not None:
                        pending.append(child)

                if any(
                    node._consumer_ownership.active_consumers
                    or node._consumer_ownership.cleanup_started
                    for node in nodes
                ):
                    raise RuntimeError(
                        "Cannot attach a continuation with active or cleaned-up "
                        "consumers."
                    )
                parent_state = self._consumer_ownership
                for node in nodes:
                    node._consumer_ownership = parent_state
                self._continuation = continuation
                continuation._is_continuation_member = True

    def _acquire_content_event_consumer(self) -> _StreamConsumerState:
        with _STREAM_CONSUMER_OWNERSHIP_LOCK:
            ownership = self._consumer_ownership
            ownership.acquire()
            return ownership

    async def _finish_content_event_consumer(
        self,
        ownership: _StreamConsumerState,
        *,
        completed: bool,
        primary_error: BaseException | None,
        await_finalizers: bool = True,
    ) -> None:
        finalizer_error = None
        if await_finalizers:
            try:
                await self._await_pending_finalizers(primary_error)
            except BaseException as error:
                finalizer_error = error

        should_cleanup = ownership.release(
            cleanup=not completed and self._get_continuation() is not None
        )
        cleanup_error = None
        if should_cleanup:
            try:
                await self._cleanup_unconsumed_continuation(
                    primary_error or finalizer_error
                )
            except BaseException as error:
                cleanup_error = error
            finally:
                ownership.mark_cleanup_complete()

        if finalizer_error is not None:
            raise finalizer_error
        if cleanup_error is not None:
            raise cleanup_error

    def _get_continuation(self) -> BaseStreamResponse | None:
        with self._finalizer_lock:
            return self._continuation

    def _adopt_continuation_state(self, continuation: BaseStreamResponse) -> None:
        self._settled_output = getattr(
            continuation, "_settled_output", continuation.data
        )
        self._final_status = continuation._final_status
        self.error = continuation.error

    async def _cleanup_unconsumed_continuation(
        self, primary_error: BaseException | None
    ) -> None:
        continuation = self._get_continuation()
        if continuation is None:
            return
        cleanup_errors = []
        if not continuation._is_finalized():
            interrupted_error = (
                primary_error
                if isinstance(primary_error, asyncio.CancelledError)
                else None
            )
            try:
                continuation.finish(error=interrupted_error, status="interrupted")
            except (asyncio.CancelledError, Exception) as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for cleanup in (
            lambda: continuation._await_pending_finalizers(primary_error),
            continuation._run_consumer_finalizers,
            lambda: continuation._cleanup_unconsumed_continuation(primary_error),
        ):
            try:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result
            except (asyncio.CancelledError, Exception) as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            if primary_error is None:
                raise cleanup_errors[0]
            for cleanup_error in cleanup_errors:
                primary_error.add_note(f"Continuation cleanup failed: {cleanup_error}")

    @staticmethod
    async def _close_consumer(iterator, primary_error: BaseException | None) -> None:
        try:
            await iterator.aclose()
        except asyncio.CancelledError as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Continuation consumer close was cancelled: {cleanup_error}"
            )
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Continuation consumer close failed: {cleanup_error}"
            )

    @staticmethod
    def _combine_finalizer_error(
        primary_error: BaseException | None,
        finalizer_error: Exception,
    ) -> BaseException:
        """Keep both ordinary failures while preserving cancellation identity."""
        if primary_error is None or primary_error is finalizer_error:
            return finalizer_error
        if isinstance(primary_error, (asyncio.CancelledError, AbortRequestedError)):
            return primary_error
        if isinstance(primary_error, Exception):
            return ExceptionGroup(
                "Stream and finalizer both failed",
                [primary_error, finalizer_error],
            )

        return primary_error

    def finish(
        self,
        *,
        error: BaseException | None = None,
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
        try:
            self._run_finalizers(status=status)
        except Exception as finalizer_error:
            combined_error = self._combine_finalizer_error(self.error, finalizer_error)
            self.set_error(combined_error)
            self._final_status = (
                "interrupted"
                if status == "interrupted"
                and isinstance(
                    self.error, (asyncio.CancelledError, AbortRequestedError)
                )
                else "failed"
            )
            if combined_error is finalizer_error:
                raise
            raise combined_error from finalizer_error
        finally:
            self._close_stream_queues()

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
            if inspect.isawaitable(result) or isinstance(result, Future):
                with self._finalizer_lock:
                    self._pending_finalizer_awaitables.append(result)

    async def _await_pending_finalizers(
        self, primary_error: BaseException | None = None
    ) -> None:
        """Finish async callbacks emitted by a synchronous stream producer."""
        with self._finalizer_lock:
            pending = list(self._pending_finalizer_awaitables)
        if not pending:
            return

        waiter = asyncio.gather(
            *[
                asyncio.wrap_future(item) if isinstance(item, Future) else item
                for item in pending
            ]
        )
        task = asyncio.current_task()
        cancellation_count = task.cancelling() if task is not None else 0
        try:
            # Unlike shield(), wait() does not add a second exception logger
            # after consumer cancellation (Python 3.14). The gather keeps running,
            # and our observer below handles its eventual failure.
            await asyncio.wait((waiter,))
            waiter.result()
        except asyncio.CancelledError as cancellation_error:
            if task is not None and task.cancelling() > cancellation_count:
                # The wait leaves finalizers running after this consumer exits.
                # Retrieve a later failure so the owned gather does not report an
                # unhandled exception when there is no waiter left to observe it.
                waiter.add_done_callback(self._observe_finalizer_group)
                if primary_error is not None:
                    raise cancellation_error from primary_error
            elif primary_error is not None and primary_error is not cancellation_error:
                raise primary_error from cancellation_error
            raise
        except Exception as finalizer_error:
            if primary_error is None:
                raise
            combined_error = self._combine_finalizer_error(
                primary_error, finalizer_error
            )
            if combined_error is finalizer_error:
                raise
            raise combined_error from finalizer_error

    @staticmethod
    def _observe_finalizer_group(waiter: asyncio.Future) -> None:
        """Retrieve a finalizer failure after its consumer has been cancelled."""
        if waiter.cancelled():
            return
        error = waiter.exception()
        if error is not None:
            logger.error(
                "Stream finalizer failed after consumer cancellation",
                exc_info=(type(error), error, error.__traceback__),
            )

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

    def add_commentary(self, data: str) -> None:
        """Publish a user-visible intermediate assistant message."""
        if not data:
            return
        self.commentary.append(data)
        if not self.first_chunk_event.is_set():
            self.first_chunk_event.set()
        self._add_event(LMStreamEvent(type="commentary.delta", data=data))

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
            if self._source_error is not None:
                raise self._source_error
            return None
        return chunk

    async def consume(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields content chunks until None is received."""
        ownership = self._acquire_content_event_consumer()
        primary_error = None
        completed = False
        try:
            while True:
                chunk = await self.next_chunk()
                if chunk is None:
                    break
                yield chunk
            await self._await_pending_finalizers()
            continuation = self._get_continuation()
            if continuation is not None or self._is_continuation_member:
                self._run_consumer_finalizers()
            if continuation is not None:
                child_stream = continuation.consume()
                child_error = None
                try:
                    async for chunk in child_stream:
                        yield chunk
                except BaseException as error:
                    child_error = error
                    self._adopt_continuation_state(continuation)
                    raise
                finally:
                    await self._close_consumer(child_stream, child_error)
                self._adopt_continuation_state(continuation)
            completed = True
        except BaseException as error:
            # GeneratorExit is normal async-generator cleanup. Let a pending
            # finalizer failure escape instead of being suppressed by aclose().
            primary_error = None if isinstance(error, GeneratorExit) else error
            raise
        finally:
            await self._finish_content_event_consumer(
                ownership, completed=completed, primary_error=primary_error
            )

    async def consume_reasoning(self) -> AsyncGenerator[str, None]:
        """Async generator that yields reasoning chunks until None is received."""
        queue = self._bind_reasoning_queue()
        while True:
            chunk = await queue.get()
            if chunk is None:
                if self._source_error is not None:
                    raise self._source_error
                break
            yield chunk
        await self._await_pending_finalizers()
        continuation = self._get_continuation()
        if continuation is not None:
            child_stream = continuation.consume_reasoning()
            child_error = None
            try:
                async for chunk in child_stream:
                    yield chunk
            except BaseException as error:
                child_error = error
                self._adopt_continuation_state(continuation)
                raise
            finally:
                await self._close_consumer(child_stream, child_error)
            self._adopt_continuation_state(continuation)

    async def consume_reasoning_summary(self) -> AsyncGenerator[str, None]:
        """Yield reasoning-summary chunks without conflating them with CoT."""
        queue = self._bind_reasoning_summary_queue()
        while True:
            chunk = await queue.get()
            if chunk is None:
                if self._source_error is not None:
                    raise self._source_error
                break
            yield chunk
        await self._await_pending_finalizers()
        continuation = self._get_continuation()
        if continuation is not None:
            child_stream = continuation.consume_reasoning_summary()
            child_error = None
            try:
                async for chunk in child_stream:
                    yield chunk
            except BaseException as error:
                child_error = error
                self._adopt_continuation_state(continuation)
                raise
            finally:
                await self._close_consumer(child_stream, child_error)
            self._adopt_continuation_state(continuation)

    async def _consume_own_events(self) -> AsyncGenerator[LMStreamEvent, None]:
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        subscriber = (loop, queue)
        with self._event_queue_lock:
            events = tuple(self._events)
            closed = self._events_closed
            if not closed:
                self._event_subscribers.append(subscriber)
        primary_error = None
        try:
            for event in events:
                yield event
            if not closed:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event
            if self._source_error is not None:
                raise self._source_error
        except BaseException as error:
            primary_error = None if isinstance(error, GeneratorExit) else error
            raise
        finally:
            await self._finish_own_event_consumption(subscriber, primary_error)

    async def _finish_own_event_consumption(self, subscriber, primary_error) -> None:
        with self._event_queue_lock:
            if subscriber in self._event_subscribers:
                self._event_subscribers.remove(subscriber)
        await self._await_pending_finalizers(primary_error)
        if primary_error is None and (
            self._get_continuation() is not None or self._is_continuation_member
        ):
            self._run_consumer_finalizers()

    async def consume_events(self) -> AsyncGenerator[LMStreamEvent, None]:
        """Yield language-model events in their original provider order."""
        ownership = self._acquire_content_event_consumer()
        continuation = None
        primary_error = None
        completed = False
        own_events = self._consume_own_events()
        try:
            async for event in own_events:
                yield event
            continuation = self._get_continuation()
            if continuation is not None:
                yield LMStreamEvent(
                    type="message.end",
                    data={"content": getattr(self, "_settled_output", self.data)},
                )
                yield LMStreamEvent(type="message.start", data={"buffered": False})
                child_events = continuation.consume_events()
                child_error = None
                try:
                    async for event in child_events:
                        yield event
                except BaseException as error:
                    child_error = error
                    self._adopt_continuation_state(continuation)
                    raise
                finally:
                    await self._close_consumer(child_events, child_error)
                self._adopt_continuation_state(continuation)
            completed = True
        except BaseException as error:
            primary_error = None if isinstance(error, GeneratorExit) else error
            raise
        finally:
            close_error = None
            try:
                await self._close_consumer(own_events, primary_error)
            except BaseException as error:
                close_error = error
            await self._finish_content_event_consumer(
                ownership,
                completed=completed,
                primary_error=primary_error or close_error,
                await_finalizers=False,
            )
            if close_error is not None:
                raise close_error
