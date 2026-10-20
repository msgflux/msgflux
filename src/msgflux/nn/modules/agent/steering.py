"""Run-scoped user input admission and final-response continuation."""

from typing import Any

import msgspec

from msgflux.exceptions import TaskPauseRequestedError, _GuardInterrupt
from msgflux.runtime.agent_inbox.base import _USER_CONTENT_SOURCES
from msgflux.runtime.events import EventType, emit_event


class _SteeringContinuation(msgspec.Struct):
    messages: Any


class AgentSteeringMixin:
    def _process_model_response(self, message, model_response, **inputs):
        while True:
            result = self._process_model_response_step(
                message, model_response, **inputs
            )
            if not isinstance(result, _SteeringContinuation):
                return result
            inputs["messages"] = result.messages
            try:
                model_response = self._execute_model(**inputs)
            except _GuardInterrupt as error:
                model_response = self._guard_model_response(error.response)

    async def _aprocess_model_response(self, message, model_response, **inputs):
        while True:
            result = await self._aprocess_model_response_step(
                message, model_response, **inputs
            )
            if not isinstance(result, _SteeringContinuation):
                return result
            inputs["messages"] = result.messages
            try:
                model_response = await self._aexecute_model(**inputs)
            except _GuardInterrupt as error:
                model_response = self._guard_model_response(error.response)

    def _emit_intermediate_answer(self, response):
        if any(
            self.has_lifecycle_hooks(event)
            for event in ("before_run_end", "after_run_end", "transform_output")
        ):
            return
        emit_event(EventType.MESSAGE_START, {"buffered": False})
        emit_event(EventType.MESSAGE_END, {"content": response})

    def _steering_stream_result(self, result):
        # The original stream already carries the public presentation envelope.
        # A linked response contributes its value, not a second nested envelope.
        if self.config.get("return_messages", False) and isinstance(result, dict):
            return result["response"]
        return result

    def _open_steering(self):
        inbox = self._get_effective_agent_inbox()
        if inbox is not None:
            inbox.open_user_messages()

    def _seal_steering(self, messages, *, force=False):
        inbox = self._get_effective_agent_inbox()
        if inbox is None:
            return True
        return inbox.close_user_messages(
            only_if_empty=not force,
            consumed_ids=self._inbox_receipt_ids(messages),
        )

    def _pending_steering_ids(self, messages):
        inbox = self._get_effective_agent_inbox()
        consumed = self._inbox_receipt_ids(messages)
        return {
            notice.notification_id
            for notice in inbox.peek()
            if notice.source in _USER_CONTENT_SOURCES
            and notice.notification_id not in consumed
        }

    def _require_steering_delivery(self, pending, messages):
        if not pending.intersection(self._inbox_receipt_ids(messages)):
            raise TaskPauseRequestedError(
                message="Accepted steering is pending but could not be delivered. "
                "Resume after releasing its inbox claim or updating notification hooks."
            )

    def _ready_to_complete(self, messages, runtime_vars, *, force=False):
        if self._seal_steering(messages, force=force):
            return True
        pending = self._pending_steering_ids(messages)
        self._drain_inbox_into_messages(messages, vars=runtime_vars)
        self._require_steering_delivery(pending, messages)
        # Persist both the preceding answer and the accepted input before asking
        # again. Existing receipt ACKs run only after this checkpoint succeeds.
        self._checkpoint_save(messages, runtime_vars)
        return False

    async def _aready_to_complete(self, messages, runtime_vars, *, force=False):
        if self._seal_steering(messages, force=force):
            return True
        pending = self._pending_steering_ids(messages)
        await self._adrain_inbox_into_messages(messages, vars=runtime_vars)
        self._require_steering_delivery(pending, messages)
        await self._acheckpoint_save(messages, runtime_vars)
        return False
