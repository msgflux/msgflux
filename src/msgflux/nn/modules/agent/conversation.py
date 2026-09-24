# ruff: noqa: A001, A002

import contextvars
import warnings
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import msgspec

from msgflux._private.response_metadata import attach_response_metadata
from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.hooks.events import (
    NotificationContext,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentNotification,
)
from msgflux.runtime.agent_run import AgentRun, get_agent_run
from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    new_run_id,
    new_thread_id,
)
from msgflux.runtime.events import EventType, emit_event
from msgflux.tools.runtime import ToolOutcome
from msgflux.utils.console import cprint
from msgflux.utils.msgspec import lossless_json_roundtrip
from msgflux.utils.time import utc_now_isoformat
from msgflux.utils.xml import apply_xml_tags

if TYPE_CHECKING:
    pass
from msgflux.nn.modules.agent.context import (
    _CURRENT_AGENT_CONTEXT,
    _require_lifecycle_payload,
)

_TASK_RESULT_UNSET = object()


class AgentConversationMixin:
    """Conversation state, inbox, checkpoint, and durable-resume behavior."""

    def _coerce_chat_messages(
        self,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]] = None,
    ) -> ChatMessages:
        if messages is None:
            return ChatMessages()
        if isinstance(messages, ChatMessages):
            return messages
        if isinstance(messages, list):
            return ChatMessages(messages)
        raise TypeError(
            "`messages` must be a `ChatMessages`, a list of mappings or None, "
            f"given `{type(messages)}`"
        )

    # --- Execution Context Resolution ---

    def _get_effective_checkpoint_store(self):
        checkpoint_store = getattr(self, "checkpoint_store", None)
        if checkpoint_store is not None:
            return checkpoint_store
        return get_execution_context().get("checkpoint_store")

    def _get_effective_task_store(self):
        return get_execution_context().get("task_store")

    def _get_effective_agent_inbox(self):
        inherited = get_execution_context().get("agent_inbox")
        if inherited is not None:
            return inherited
        return getattr(self, "agent_inbox", None)

    def _get_scoped_agent_inbox(self, scope: ExecutionScope):
        inherited = get_execution_context().get("agent_inbox")
        namespace = self.get_module_name()
        if inherited is not None:
            return inherited.fork(
                owner=namespace,
                namespace=namespace,
                thread_id=scope.thread_id,
                run_id=scope.run_id,
            )
        inbox = getattr(self, "agent_inbox", None)
        if inbox is None:
            return None
        inbox.bind_scope(scope, namespace=namespace)
        return inbox.fork(
            owner=namespace,
            namespace=namespace,
            thread_id=scope.thread_id,
            run_id=scope.run_id,
        )

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self.agent_inbox = agent_inbox
        self.tool_library.set_agent_inbox(agent_inbox)

    def _raise_if_background_task_interrupted(self) -> None:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is None:
            return
        if task_handle.is_interrupt_requested():
            if self.config.get("verbose", False):
                cprint(
                    f"[{self.name}][task_interrupt] task_id={task_handle.task_id}",
                    bc="b",
                    ls="b",
                )
            raise TaskInterruptRequestedError(task_handle.task_id)

    def _handle_control_notifications(
        self,
        notifications: List[AgentNotification],
    ) -> List[AgentNotification]:
        remaining = []
        for notification in notifications:
            if notification.source != "control":
                remaining.append(notification)
                continue

            command = (notification.status or "").lower()
            reason = notification.metadata.get("reason")
            task_handle = get_execution_context().get("task_handle")
            task_id = getattr(task_handle, "task_id", None)

            if command == "interrupt":
                raise TaskInterruptRequestedError(
                    task_id or get_execution_context().get("run_id") or "unknown",
                    str(reason) if reason else None,
                )
            if command == "pause":
                raise TaskPauseRequestedError(
                    task_id if isinstance(task_id, str) else None,
                    str(reason) if reason else None,
                )

            remaining.append(notification)
        return remaining

    @staticmethod
    def _inbox_receipt_ids(messages) -> set[str]:
        if isinstance(messages, ChatMessages):
            return set(messages.metadata.get("inbox_receipts", ()))
        return {
            receipt
            for item in messages
            for receipt in item.get("metadata", {}).get("inbox_receipts", ())
        }

    def _record_inbox_receipts(self, messages, ids) -> None:
        if isinstance(messages, ChatMessages):
            messages.metadata["inbox_receipts"] = sorted(
                self._inbox_receipt_ids(messages).union(ids)
            )

    @staticmethod
    def _emit_notification_drain(notification_ids) -> None:
        ids = sorted(set(notification_ids))
        if ids:
            emit_event(
                EventType.NOTIFICATION_DRAIN,
                {"count": len(ids), "notification_ids": ids},
            )

    @staticmethod
    def _forward_task_messages(inbox: AgentInbox) -> None:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is not None:
            task_handle.forward_messages(inbox)

    @staticmethod
    def _ack_task_messages(notification_ids) -> None:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is not None:
            task_handle.ack_messages(list(notification_ids))

    def _prepare_inbox_delivery(self, inbox, messages, notifications, *, drain):
        if not drain:
            return self._handle_control_notifications(notifications)
        known = self._inbox_receipt_ids(messages)
        inbox.mark_delivered(
            item.notification_id
            for item in notifications
            if item.notification_id in known
        )
        pending = [item for item in notifications if item.notification_id not in known]
        try:
            return self._handle_control_notifications(pending)
        except (TaskInterruptRequestedError, TaskPauseRequestedError) as error:
            command = (
                "interrupt"
                if isinstance(error, TaskInterruptRequestedError)
                else "pause"
            )
            consumed = next(
                (
                    item.notification_id
                    for item in pending
                    if item.source == "control" and item.status == command
                ),
                None,
            )
            if consumed is not None:
                self._record_inbox_receipts(messages, [consumed])
                inbox.mark_delivered([consumed])
            inbox.release(except_ids=inbox.delivered_ids())
            if self._get_effective_checkpoint_store() is None:
                inbox.ack(inbox.delivered_ids())
            raise
        except BaseException:
            inbox.release()
            raise

    def _finish_inbox_delivery(self, inbox, messages, notifications, *, drain):
        ids = [item.notification_id for item in notifications]
        notification_messages = inbox.render_messages(notifications)
        for item in notification_messages:
            item["metadata"] = {**item.get("metadata", {}), "inbox_receipts": ids}
        self._persist_notification_messages(messages, notification_messages)
        if drain:
            self._record_inbox_receipts(messages, ids)
            inbox.mark_delivered(ids)
            inbox.release(except_ids=inbox.delivered_ids())
            if self._get_effective_checkpoint_store() is None:
                delivered_ids = inbox.delivered_ids()
                inbox.ack(delivered_ids)
                self._ack_task_messages(delivered_ids)
                self._emit_notification_drain(delivered_ids)
        return bool(notification_messages)

    def _drain_inbox_into_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        vars: Optional[Mapping[str, Any]] = None,
        scope: Optional[ExecutionScope] = None,
        drain_notifications: bool = True,
    ) -> bool:
        inbox = self._get_effective_agent_inbox()
        if inbox is None:
            return False

        self._forward_task_messages(inbox)

        notifications = inbox.claim() if drain_notifications else inbox.peek()
        notifications = self._prepare_inbox_delivery(
            inbox, messages, notifications, drain=drain_notifications
        )
        if not notifications:
            self._finish_inbox_delivery(inbox, messages, [], drain=drain_notifications)
            return False

        try:
            notification_context = self._run_lifecycle_hooks(
                "transform_notifications",
                NotificationContext(
                    scope=scope or get_execution_context()["scope"],
                    vars=vars or {},
                    notifications=tuple(notifications),
                    messages=messages,
                ),
            )
            notification_context = _require_lifecycle_payload(
                "transform_notifications", notification_context, NotificationContext
            )
            notifications = list(notification_context.notifications)
            if not all(isinstance(item, AgentNotification) for item in notifications):
                raise TypeError(
                    "NotificationContext.notifications must contain AgentNotification"
                )
            return self._finish_inbox_delivery(
                inbox, messages, notifications, drain=drain_notifications
            )
        except BaseException:
            if drain_notifications:
                inbox.release()
            raise

    async def _adrain_inbox_into_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        vars: Optional[Mapping[str, Any]] = None,
        scope: Optional[ExecutionScope] = None,
        drain_notifications: bool = True,
    ) -> bool:
        inbox = self._get_effective_agent_inbox()
        if inbox is None:
            return False

        self._forward_task_messages(inbox)

        notifications = inbox.claim() if drain_notifications else inbox.peek()
        notifications = self._prepare_inbox_delivery(
            inbox, messages, notifications, drain=drain_notifications
        )
        if not notifications:
            self._finish_inbox_delivery(inbox, messages, [], drain=drain_notifications)
            return False

        try:
            notification_context = await self._arun_lifecycle_hooks(
                "transform_notifications",
                NotificationContext(
                    scope=scope or get_execution_context()["scope"],
                    vars=vars or {},
                    notifications=tuple(notifications),
                    messages=messages,
                ),
            )
            notification_context = _require_lifecycle_payload(
                "transform_notifications", notification_context, NotificationContext
            )
            notifications = list(notification_context.notifications)
            if not all(isinstance(item, AgentNotification) for item in notifications):
                raise TypeError(
                    "NotificationContext.notifications must contain AgentNotification"
                )
            return self._finish_inbox_delivery(
                inbox, messages, notifications, drain=drain_notifications
            )
        except BaseException:
            if drain_notifications:
                inbox.release()
            raise

    # --- Inbox Delivery ---

    def _build_model_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        vars: Optional[Mapping[str, Any]] = None,
        scope: Optional[ExecutionScope] = None,
        drain_notifications: bool = True,
    ) -> Union[ChatMessages, List[Mapping[str, Any]]]:
        if isinstance(messages, ChatMessages):
            working_messages: Union[ChatMessages, List[Mapping[str, Any]]] = (
                messages if drain_notifications else messages.copy()
            )
        else:
            working_messages = messages if drain_notifications else list(messages)

        self._drain_inbox_into_messages(
            working_messages,
            vars=vars,
            scope=scope,
            drain_notifications=drain_notifications,
        )
        return self._project_task_context(working_messages)

    async def _abuild_model_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        vars: Optional[Mapping[str, Any]] = None,
        scope: Optional[ExecutionScope] = None,
        drain_notifications: bool = True,
    ) -> Union[ChatMessages, List[Mapping[str, Any]]]:
        if isinstance(messages, ChatMessages):
            working_messages: Union[ChatMessages, List[Mapping[str, Any]]] = (
                messages if drain_notifications else messages.copy()
            )
        else:
            working_messages = messages if drain_notifications else list(messages)

        await self._adrain_inbox_into_messages(
            working_messages,
            vars=vars,
            scope=scope,
            drain_notifications=drain_notifications,
        )
        return self._project_task_context(working_messages)

    def _project_task_context(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
    ) -> Union[ChatMessages, List[Mapping[str, Any]]]:
        """Render stored task context only in the model-facing projection."""
        has_context = any(
            isinstance(item.get("metadata"), Mapping)
            and item["metadata"].get("task_context")
            for item in messages
        )
        if not has_context:
            return messages

        projected = (
            messages.copy()
            if isinstance(messages, ChatMessages)
            else [deepcopy(dict(item)) for item in messages]
        )
        for item in projected:
            metadata = item.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            task_context = metadata.get("task_context")
            if not task_context:
                continue
            projected_metadata = dict(metadata)
            projected_metadata.pop("task_context", None)
            if projected_metadata:
                item["metadata"] = projected_metadata
            else:
                item.pop("metadata", None)
            prefix = apply_xml_tags("context", str(task_context)) + "\n\n"
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = prefix + content
                continue
            if isinstance(content, list):
                text_parts = [
                    part
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                if text_parts:
                    text_parts[-1]["text"] = prefix + str(
                        text_parts[-1].get("text", "")
                    )
                else:
                    content.append({"type": "text", "text": prefix.rstrip()})
        return projected

    def _persist_notification_message(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        notification_message: Optional[Mapping[str, Any]],
    ) -> None:
        if notification_message is None:
            return
        if isinstance(messages, ChatMessages):
            is_conversation_content = "inbox_origin" in notification_message.get(
                "metadata", {}
            ) or isinstance(notification_message.get("content"), list)
            if messages.get_active_turn_size() <= 2 and not is_conversation_content:
                messages.insert_before_active_turn(notification_message)
            else:
                messages.append(notification_message)
            return
        messages.append(notification_message)

    def _persist_notification_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        notification_messages: List[Mapping[str, Any]],
    ) -> None:
        for notification_message in notification_messages:
            self._persist_notification_message(messages, notification_message)

    # --- Thread And Run Resolution ---

    def _prepare_messages_scope(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        scope: Optional[ExecutionScope],
    ) -> Tuple[
        Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        ExecutionScope,
        str,
        str,
    ]:
        effective_checkpoint_store = self._get_effective_checkpoint_store()
        should_use_chat_messages = (
            effective_checkpoint_store is not None
            or isinstance(messages, ChatMessages)
            or scope is not None
            or self.tool_library.has_deferred_tools
        )
        if should_use_chat_messages:
            messages = self._coerce_chat_messages(messages)

        effective_thread_id = self._resolve_thread_id(
            messages=messages,
            thread_id=scope.thread_id if scope is not None else None,
        )
        effective_run_id = self._resolve_run_id(
            messages=messages,
            run_id=scope.run_id if scope is not None else None,
        )
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            thread_id=effective_thread_id,
            namespace=self.get_module_name(),
            run_id=effective_run_id,
        )
        return messages, effective_scope, effective_thread_id, effective_run_id

    def _resolve_thread_id(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        thread_id: Optional[str],
    ) -> str:
        if isinstance(thread_id, str) and thread_id:
            return thread_id
        if isinstance(messages, ChatMessages) and messages.thread_id:
            return messages.thread_id
        inherited = get_execution_context().get("thread_id")
        if isinstance(inherited, str) and inherited:
            return inherited
        return new_thread_id()

    def _resolve_run_id(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        run_id: Optional[str],
    ) -> str:
        if isinstance(run_id, str) and run_id:
            return run_id
        if isinstance(messages, ChatMessages):
            active_turn = messages.get_active_turn()
            if active_turn and isinstance(active_turn.get("turn_id"), str):
                return active_turn["turn_id"]
        inherited = get_execution_context().get("run_id")
        if isinstance(inherited, str) and inherited:
            return inherited
        return new_run_id()

    # --- Chat Turn Tracking ---

    def _start_chat_turn_if_needed(
        self,
        *,
        messages: ChatMessages,
        turn_id: str,
    ) -> None:
        messages._ensure_stream_available()
        active_turn = messages.get_active_turn()
        if active_turn is not None:
            if active_turn.get("turn_id") == turn_id:
                return
            messages.end_turn(event="interrupt")

        messages.begin_turn(
            namespace=self.get_module_name(),
            turn_id=turn_id,
        )

    def _append_response_to_chat_messages(  # noqa: C901
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        metadata: Optional[Mapping[str, Any]],  # noqa: ARG002
        *,
        reasoning: str | None = None,
        history_items: Optional[List[Mapping[str, Any]]] = None,
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return
        if isinstance(raw_response, ModelStreamResponse):
            return
        if response_type != "text_generation" and "structured" not in response_type:
            return

        if history_items:
            messages.extend(history_items)
            if any(item.get("type") == "reasoning" for item in history_items):
                reasoning = None
            if any(
                item.get("type") == "message" and item.get("role") == "assistant"
                for item in history_items
            ):
                return

        answer = None
        reasoning_content = reasoning
        if isinstance(raw_response, str):
            answer = raw_response
        elif isinstance(raw_response, Mapping):
            answer = raw_response.get("answer")
            if answer is None and "answer" not in raw_response:
                answer = raw_response.get("text")
            reasoning_content = (
                self._extract_reasoning_content(raw_response) or reasoning_content
            )
        elif raw_response is not None:
            answer = str(raw_response)

        if reasoning_content is not None or answer is not None:
            messages.add_assistant_response(
                content=answer,
                reasoning_content=reasoning_content,
            )

    # --- Response Extraction Helpers ---

    def _append_tool_model_history(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        model_response: Union[ModelResponse, ModelStreamResponse],
    ) -> set[str]:
        uses_canonical_history = isinstance(
            messages, ChatMessages
        ) or self._model_uses_canonical_history(messages)
        if not uses_canonical_history:
            return set()
        if isinstance(model_response, ModelStreamResponse):
            items = model_response.chat_accumulator.snapshot(
                fallback_reasoning=model_response.reasoning
            )
        else:
            items = getattr(model_response, "history_items", [])
        if not isinstance(items, list):
            return set()
        trajectory_items = [
            item
            for item in items
            if item.get("type")
            in {
                "reasoning",
                "tool_search_call",
                "tool_search_output",
                "function_call",
            }
        ]
        messages.extend(trajectory_items)

        existing_call_ids = {
            item.get("call_id") or item.get("id")
            for item in messages
            if item.get("type") == "function_call"
        }
        get_tool_intents = getattr(model_response, "get_tool_intents", None)
        if callable(get_tool_intents):
            missing_calls = []
            for intent in get_tool_intents():
                if intent.id in existing_call_ids:
                    continue
                arguments = intent.arguments
                if isinstance(arguments, str):
                    serialized_arguments = arguments
                else:
                    serialized_arguments = msgspec.json.encode(
                        arguments if arguments is not None else {}
                    ).decode()
                missing_calls.append(
                    {
                        "type": "function_call",
                        "call_id": intent.id,
                        "name": intent.name,
                        "arguments": serialized_arguments,
                    }
                )
                existing_call_ids.add(intent.id)
            messages.extend(missing_calls)
            trajectory_items.extend(missing_calls)

        return {item["type"] for item in trajectory_items}

    def _extend_tool_response_history(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        tool_response_messages: List[Mapping[str, Any]],
    ) -> None:
        uses_canonical_history = isinstance(
            messages, ChatMessages
        ) or self._model_uses_canonical_history(messages)
        if not uses_canonical_history:
            messages.extend(tool_response_messages)
            return

        existing_call_ids = {
            item.get("call_id")
            for item in messages
            if item.get("type") == "function_call"
        }
        normalized = ChatMessages(tool_response_messages).to_items()
        messages.extend(
            item
            for item in normalized
            if not (
                item.get("type") == "function_call"
                and item.get("call_id") in existing_call_ids
            )
        )

    def _model_uses_canonical_history(self, messages) -> bool:
        if not isinstance(messages, list):
            return False
        try:
            model = self.model
        except AttributeError:
            return False
        return bool(getattr(model, "_uses_canonical_history", False))

    def _extract_reasoning_content(
        self,
        payload: Mapping[str, Any],
    ) -> Optional[str]:
        for field in ("reasoning_content", "reasoning_text", "think", "reasoning"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                return value
        return None

    def _finalize_chat_turn(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return

        if isinstance(raw_response, ModelStreamResponse):
            return
        messages.end_turn(event="complete")

    def _attach_stream_checkpoint_finalizer(  # noqa: C901
        self,
        model_response: ModelStreamResponse,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        *,
        async_mode: bool = False,
    ) -> None:
        # Producers can finish on a worker thread after Agent.forward has returned.
        # Capture the execution identity while it is still bound.
        context = contextvars.copy_context()
        scope = get_execution_context().get("scope")
        tracked = isinstance(messages, ChatMessages)
        source_state = messages._to_state() if tracked else deepcopy(messages)
        stream_messages = messages.copy() if tracked else deepcopy(messages)
        run_id = None
        if tracked:
            active_turn = messages.get_active_turn()
            if active_turn is None or not isinstance(active_turn.get("turn_id"), str):
                raise RuntimeError(
                    "Cannot attach a stream without an active Agent turn"
                )
            run_id = active_turn["turn_id"]
            messages._claim_stream(run_id)

        def prepare(final_state):
            start = len(stream_messages)
            stream_messages.extend(final_state.items)
            attach_response_metadata(
                stream_messages, final_state.metadata, after_index=start
            )
            return self._run_end_context(
                outcome=final_state.status,
                messages=stream_messages,
                vars=vars,
                scope=scope,
                output=final_state.output,
                error=final_state.error,
            )

        def close_turn(settled, status, error):
            if not isinstance(settled, ChatMessages):
                return
            if status == "interrupted":
                self._close_interrupted_tool_calls(
                    settled, reason=str(error) if error is not None else None
                )
            elif settled.get_active_turn() is not None:
                settled.end_turn(event="complete" if status == "completed" else "fail")

        def release(settled, committed):
            if not tracked:
                if committed:
                    messages[:] = settled
                return
            try:
                if committed:
                    if messages._to_state() != source_state:
                        raise RuntimeError(
                            "ChatMessages changed while its Agent stream was active; "
                            "the completed stream was checkpointed but was not allowed "
                            "to overwrite the newer in-memory history."
                        )
                    messages._hydrate_state(settled._to_state())
            finally:
                messages._release_stream(run_id)

        def finalize_stream(final_state):
            settled = stream_messages
            committed = False
            try:
                run_end = prepare(final_state)
                try:
                    run_end = self._run_run_end_hook("before_run_end", run_end)
                except Exception:
                    close_turn(settled, "failed", final_state.error)
                    self._checkpoint_save(settled, vars, status="failed")
                    committed = True
                    raise
                settled = run_end.messages
                close_turn(settled, final_state.status, final_state.error)
                self._checkpoint_save(
                    settled,
                    vars,
                    status=final_state.status,
                    task_result=run_end.output,
                )
                committed = True
                run_end = self._run_after_run_end_hook(run_end)
                model_response._settled_output = run_end.output
            finally:
                release(settled, committed)

        async def finalize_async(final_state):
            settled = stream_messages
            committed = False
            try:
                run_end = prepare(final_state)
                try:
                    run_end = await self._arun_run_end_hook("before_run_end", run_end)
                except Exception:
                    close_turn(settled, "failed", final_state.error)
                    await self._acheckpoint_save(settled, vars, status="failed")
                    committed = True
                    raise
                settled = run_end.messages
                close_turn(settled, final_state.status, final_state.error)
                await self._acheckpoint_save(
                    settled,
                    vars,
                    status=final_state.status,
                    task_result=run_end.output,
                )
                committed = True
                run_end = await self._arun_after_run_end_hook(run_end)
                model_response._settled_output = run_end.output
            finally:
                release(settled, committed)

        if async_mode:
            model_response.add_finalizer(finalize_async)
        else:
            model_response.add_finalizer(
                lambda state: context.run(finalize_stream, state)
            )

    def _close_interrupted_tool_calls(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]], None],
        *,
        reason: str | None = None,
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return
        messages.close_interrupted_tool_calls(reason=reason)
        if messages.get_active_turn() is not None:
            messages.end_turn(
                event="interrupt",
                metadata={"reason": reason} if reason else None,
            )

    def _append_interrupted_tool_response_messages(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        model_response: Union[ModelResponse, ModelStreamResponse],
        *,
        reason: str | None = None,
    ) -> None:
        intents = model_response.get_tool_intents()
        if not intents:
            self._close_interrupted_tool_calls(messages, reason=reason)
            return

        interrupted_outcomes = tuple(
            ToolOutcome.failed(
                intent,
                status="interrupted",
                code="tool_interrupted",
                message=reason or "Tool call interrupted.",
            )
            for intent in intents
        )
        tool_response_messages = model_response.render_tool_outcomes(
            interrupted_outcomes
        )
        self._extend_tool_response_history(messages, tool_response_messages)
        self._close_interrupted_tool_calls(messages, reason=reason)

    # --- Checkpoint Persistence ---

    def _ack_inbox_notifications(self, messages) -> None:
        inbox = self._get_effective_agent_inbox()
        if inbox is not None:
            acknowledged_ids = inbox.delivered_ids() & self._inbox_receipt_ids(messages)
            try:
                inbox.ack(acknowledged_ids)
                self._ack_task_messages(acknowledged_ids)
                self._emit_notification_drain(acknowledged_ids)
            except Exception as error:
                # The durable receipt makes replay safe. Do not turn a committed
                # terminal run into a failed run because inbox cleanup failed.
                warnings.warn(
                    f"Inbox acknowledgement failed after checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                inbox.release()
            except Exception as error:
                warnings.warn(
                    f"Inbox lease cleanup failed after checkpoint: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _release_inbox_notifications(self) -> None:
        inbox = self._get_effective_agent_inbox()
        if inbox is not None:
            inbox.release()

    def _checkpoint_save(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]], None],
        _vars: Mapping[str, Any],
        status: str = "running",
        *,
        task_result: Any = _TASK_RESULT_UNSET,
    ) -> None:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None or not isinstance(messages, ChatMessages):
            return

        turns = messages.turns
        if not turns:
            return

        thread_id = messages.thread_id or new_thread_id()
        run_id = turns[-1]["turn_id"]
        state = self._build_checkpoint_state(
            messages, status=status, task_result=task_result
        )
        try:
            run = get_agent_run()
            if run is not None and getattr(
                checkpoint_store, "supports_atomic_commit", False
            ):
                committed = checkpoint_store.commit_state(
                    self.get_module_name(),
                    thread_id,
                    run_id,
                    state,
                    expected_revision=run.revision,
                    extension_state=run.extension_state,
                    event={"event_type": "checkpoint", "status": status},
                    branch_id=run.branch_id,
                    head_item_id=run.head_item_id,
                )
                run.revision = committed.revision
            else:
                checkpoint_store.save_state(
                    self.get_module_name(), thread_id, run_id, state
                )
        except BaseException:
            self._release_inbox_notifications()
            raise
        self._ack_inbox_notifications(messages)

    async def _acheckpoint_save(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]], None],
        _vars: Mapping[str, Any],
        status: str = "running",
        *,
        task_result: Any = _TASK_RESULT_UNSET,
    ) -> None:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None or not isinstance(messages, ChatMessages):
            return

        turns = messages.turns
        if not turns:
            return

        thread_id = messages.thread_id or new_thread_id()
        run_id = turns[-1]["turn_id"]
        state = self._build_checkpoint_state(
            messages, status=status, task_result=task_result
        )
        try:
            run = get_agent_run()
            if run is not None and getattr(
                checkpoint_store, "supports_atomic_commit", False
            ):
                params = {
                    "expected_revision": run.revision,
                    "extension_state": run.extension_state,
                    "event": {"event_type": "checkpoint", "status": status},
                    "branch_id": run.branch_id,
                    "head_item_id": run.head_item_id,
                }
                if hasattr(checkpoint_store, "acommit_state"):
                    committed = await checkpoint_store.acommit_state(
                        self.get_module_name(), thread_id, run_id, state, **params
                    )
                else:
                    committed = checkpoint_store.commit_state(
                        self.get_module_name(), thread_id, run_id, state, **params
                    )
                run.revision = committed.revision
            elif hasattr(checkpoint_store, "asave_state"):
                await checkpoint_store.asave_state(
                    self.get_module_name(),
                    thread_id,
                    run_id,
                    state,
                )
            else:
                checkpoint_store.save_state(
                    self.get_module_name(), thread_id, run_id, state
                )
        except BaseException:
            self._release_inbox_notifications()
            raise
        self._ack_inbox_notifications(messages)

    def _checkpoint_interrupted(
        self,
        inputs: Mapping[str, Any],
        exc: BaseException,
    ) -> None:
        self._close_interrupted_tool_calls(
            inputs.get("messages"),
            reason=str(exc),
        )
        self._checkpoint_save(
            inputs.get("messages"),
            inputs.get("vars", {}),
            status="interrupted",
        )

    async def _acheckpoint_interrupted(
        self,
        inputs: Mapping[str, Any],
        exc: BaseException,
    ) -> None:
        self._close_interrupted_tool_calls(
            inputs.get("messages"),
            reason=str(exc),
        )
        await self._acheckpoint_save(
            inputs.get("messages"),
            inputs.get("vars", {}),
            status="interrupted",
        )

    @staticmethod
    def _raise_interrupted_from_abort(
        inputs: Mapping[str, Any],
        exc: BaseException,
    ) -> None:
        if isinstance(exc, AbortRequestedError):
            scope = inputs.get("scope")
            raise TaskInterruptRequestedError(
                scope.run_id if scope is not None else "unknown",
                str(exc),
            ) from exc
        raise exc

    def _build_checkpoint_state(
        self,
        messages: ChatMessages,
        *,
        status: str,
        task_result: Any = _TASK_RESULT_UNSET,
    ) -> Mapping[str, Any]:
        run = get_agent_run()
        context = (_CURRENT_AGENT_CONTEXT.get() or {}).get(id(self), {})
        scope = context.get("scope") or get_execution_context()["scope"]
        if run is not None:
            run.head_item_id = messages[-1].get("item_id") if messages else None
        state = {
            "schema_version": 1,
            "status": status,
            "messages": messages._to_state(),
            "runtime": run.durable_state() if run is not None else {},
            "scope": scope.to_dict(),
            "model_preference": context.get("model_preference"),
            "metadata": {
                "namespace": self.get_module_name(),
                "saved_at": utc_now_isoformat(),
            },
        }
        task_handle = get_execution_context().get("task_handle")
        if (
            status == "completed"
            and task_result is not _TASK_RESULT_UNSET
            and getattr(task_handle, "task_id", None) == scope.run_id
        ):
            lossless, value = lossless_json_roundtrip(task_result)
            if lossless:
                state["task_result"] = {"value": value}
        return state

    def _checkpoint_save_on_error(self, inputs: Mapping[str, Any]) -> None:
        if self._get_effective_checkpoint_store() is None:
            return
        messages = inputs.get("messages")
        vars = inputs.get("vars", {})
        if (
            isinstance(messages, ChatMessages)
            and messages.get_active_turn() is not None
        ):
            messages.end_turn(event="fail")
        self._checkpoint_save(messages, vars, status="failed")

    async def _acheckpoint_save_on_error(self, inputs: Mapping[str, Any]) -> None:
        if self._get_effective_checkpoint_store() is None:
            return
        messages = inputs.get("messages")
        vars = inputs.get("vars", {})
        if (
            isinstance(messages, ChatMessages)
            and messages.get_active_turn() is not None
        ):
            messages.end_turn(event="fail")
        await self._acheckpoint_save(messages, vars, status="failed")

    # --- Checkpoint Resume ---

    def _continue_thread_from_checkpoint(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        vars: Mapping[str, Any],
        model_preference: Optional[Union[str, List[str]]],
        thread_id: str,
        run_id: str,
    ) -> Tuple[
        Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        Mapping[str, Any],
        Optional[Union[str, List[str]]],
    ]:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None or not isinstance(messages, ChatMessages):
            return messages, vars, model_preference
        if messages:
            return messages, vars, model_preference

        namespace = self.get_module_name()
        if checkpoint_store.load_state(namespace, thread_id, run_id) is not None:
            return messages, vars, model_preference

        latest = checkpoint_store.load_latest_run(namespace, thread_id)
        if latest is None:
            return messages, vars, model_preference

        restored = ChatMessages()
        restored._hydrate_state(latest.get("messages", {}))
        restored.configure_thread(thread_id=thread_id, namespace=namespace)

        restored_model_preference = (
            model_preference
            if model_preference is not None
            else latest.get("model_preference")
        )
        return restored, vars, restored_model_preference

    async def _acontinue_thread_from_checkpoint(
        self,
        *,
        messages: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        vars: Mapping[str, Any],
        model_preference: Optional[Union[str, List[str]]],
        thread_id: str,
        run_id: str,
    ) -> Tuple[
        Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        Mapping[str, Any],
        Optional[Union[str, List[str]]],
    ]:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None or not isinstance(messages, ChatMessages):
            return messages, vars, model_preference
        if messages:
            return messages, vars, model_preference

        namespace = self.get_module_name()
        if hasattr(checkpoint_store, "aload_state"):
            current = await checkpoint_store.aload_state(namespace, thread_id, run_id)
        else:
            current = checkpoint_store.load_state(namespace, thread_id, run_id)
        if current is not None:
            return messages, vars, model_preference

        if hasattr(checkpoint_store, "aload_latest_run"):
            latest = await checkpoint_store.aload_latest_run(namespace, thread_id)
        else:
            latest = checkpoint_store.load_latest_run(namespace, thread_id)
        if latest is None:
            return messages, vars, model_preference

        restored = ChatMessages()
        restored._hydrate_state(latest.get("messages", {}))
        restored.configure_thread(thread_id=thread_id, namespace=namespace)

        restored_model_preference = (
            model_preference
            if model_preference is not None
            else latest.get("model_preference")
        )
        return restored, vars, restored_model_preference

    def _try_resume_from_checkpoint(
        self,
        messages_kwarg: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        *,
        scope: Optional[ExecutionScope] = None,
    ) -> Optional[Mapping[str, Any]]:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None:
            return None
        run_id = scope.run_id if scope is not None else None
        if not isinstance(run_id, str) or not run_id:
            return None

        effective_thread_id = self._resolve_thread_id(
            messages=messages_kwarg,
            thread_id=scope.thread_id if scope is not None else None,
        )
        state = checkpoint_store.load_state(
            self.get_module_name(),
            effective_thread_id,
            run_id,
        )
        if state is None:
            return None
        if state.get("status") in {"completed", "interrupted"}:
            raise ValueError(
                f"Run `{run_id}` already reached terminal status "
                f"`{state.get('status')}`. Use a new run_id to continue thread "
                f"`{effective_thread_id}`."
            )

        self._restore_agent_run(state, effective_thread_id, run_id)
        restored = ChatMessages()
        restored._hydrate_state(state.get("messages", {}))
        if not restored.turns or restored.turns[-1]["turn_id"] != run_id:
            restored.begin_turn(turn_id=run_id)
        elif restored.get_active_turn() is None:
            restored.resume_turn(run_id, metadata={"source": "checkpoint"})
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            thread_id=effective_thread_id,
            namespace=self.get_module_name(),
            run_id=run_id,
            parent_run_id=state.get("scope", {}).get("parent_run_id"),
            root_run_id=state.get("scope", {}).get("root_run_id"),
        )
        return {
            "messages": restored,
            "model_preference": state.get("model_preference"),
            "scope": effective_scope,
        }

    async def _atry_resume_from_checkpoint(
        self,
        messages_kwarg: Optional[Union[ChatMessages, List[Mapping[str, Any]]]],
        *,
        scope: Optional[ExecutionScope] = None,
    ) -> Optional[Mapping[str, Any]]:
        checkpoint_store = self._get_effective_checkpoint_store()
        if checkpoint_store is None:
            return None
        run_id = scope.run_id if scope is not None else None
        if not isinstance(run_id, str) or not run_id:
            return None

        effective_thread_id = self._resolve_thread_id(
            messages=messages_kwarg,
            thread_id=scope.thread_id if scope is not None else None,
        )
        if hasattr(checkpoint_store, "aload_state"):
            state = await checkpoint_store.aload_state(
                self.get_module_name(),
                effective_thread_id,
                run_id,
            )
        else:
            state = checkpoint_store.load_state(
                self.get_module_name(),
                effective_thread_id,
                run_id,
            )

        if state is None:
            return None
        if state.get("status") in {"completed", "interrupted"}:
            raise ValueError(
                f"Run `{run_id}` already reached terminal status "
                f"`{state.get('status')}`. Use a new run_id to continue thread "
                f"`{effective_thread_id}`."
            )

        self._restore_agent_run(state, effective_thread_id, run_id)
        restored = ChatMessages()
        restored._hydrate_state(state.get("messages", {}))
        if not restored.turns or restored.turns[-1]["turn_id"] != run_id:
            restored.begin_turn(turn_id=run_id)
        elif restored.get_active_turn() is None:
            restored.resume_turn(run_id, metadata={"source": "checkpoint"})
        effective_scope = (scope or get_execution_context()["scope"]).with_overrides(
            thread_id=effective_thread_id,
            namespace=self.get_module_name(),
            run_id=run_id,
            parent_run_id=state.get("scope", {}).get("parent_run_id"),
            root_run_id=state.get("scope", {}).get("root_run_id"),
        )
        return {
            "messages": restored,
            "model_preference": state.get("model_preference"),
            "scope": effective_scope,
        }

    def _restore_agent_run(self, state, thread_id, run_id) -> None:
        current = get_agent_run()
        if current is None:
            return
        restored = AgentRun.from_durable_state(
            state.get("runtime"),
            namespace=self.get_module_name(),
            thread_id=thread_id,
            run_id=run_id,
        )
        restored.revision = state.get("_checkpoint", {}).get("revision", 0)
        restored.namespace = self.get_module_name()
        restored.thread_id = thread_id
        restored.run_id = run_id
        current.__dict__.update(restored.__dict__)

    # --- Configuration ---
