"""Checkpointed approval batches, replayed before the next model request."""

# ruff: noqa: A002

import asyncio
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy

from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.tool_transport import validate_native_calls
from msgflux.nn.modules.agent.context import _UNSET
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.approvals.agent import (
    AgentApprovals,
    ApprovalBatch,
    ApprovalReconciliationRequiredError,
    approval_batch_active,
)
from msgflux.runtime.approvals.reconciliation import inspect_batch, reconcile_batch
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.events import EventType, _hub_event_sink
from msgflux.utils.msgspec import msgspec_dumps

_KEY = "pending_approvals"
_APPROVAL_POLICIES = ContextVar("msgflux_agent_approval_policies", default=None)


class AgentApprovalMixin:
    def _get_effective_approvals(self, approvals=_UNSET):
        if approvals is _UNSET:
            approvals = (_APPROVAL_POLICIES.get() or {}).get(id(self), self.approvals)
        if approvals is not None and not isinstance(approvals, AgentApprovals):
            raise TypeError("approvals must be AgentApprovals or None")
        return approvals

    @contextmanager
    def _approval_context(self, approvals):
        policies = dict(_APPROVAL_POLICIES.get() or {})
        policies[id(self)] = self._get_effective_approvals(approvals)
        token = _APPROVAL_POLICIES.set(policies)
        try:
            yield
        finally:
            _APPROVAL_POLICIES.reset(token)

    def inspect_approval_batch(self, thread_id: str, run_id: str):
        """Return a detached checkpoint for authenticated host reconciliation."""
        return inspect_batch(
            self._get_effective_checkpoint_store(),
            self.get_module_name(),
            thread_id,
            run_id,
        )

    def reconcile_approval_batch(self, thread_id: str, run_id: str, **decision):
        """Resolve an uncertain batch after the host has stopped its worker."""
        return reconcile_batch(
            self._get_effective_checkpoint_store(),
            self.get_module_name(),
            thread_id,
            run_id,
            **decision,
        )

    async def ainspect_approval_batch(self, thread_id: str, run_id: str):
        return await asyncio.to_thread(self.inspect_approval_batch, thread_id, run_id)

    async def areconcile_approval_batch(self, thread_id: str, run_id: str, **decision):
        return await asyncio.to_thread(
            self.reconcile_approval_batch, thread_id, run_id, **decision
        )

    def _validate_approval_execution(self, _intents):
        if self._get_effective_approvals() is not None and not approval_batch_active():
            raise ValueError("Approval policies require canonical Agent tool calls")

    def decide_approval(
        self, request_id: str, *, approved: bool, decided_by: str, approvals=_UNSET
    ):
        """Record an authenticated host decision without automatically resuming."""
        policy = self._get_effective_approvals(approvals)
        if policy is None:
            raise ValueError("Agent has no approval configuration")
        record = policy.store.decide(
            self.get_module_name(), request_id, approved=approved, decided_by=decided_by
        )
        _hub_event_sink().emit(
            EventType.TOOL_APPROVAL_RESOLVED,
            {
                "request_id": request_id,
                "status": record.status,
                "tool_call_id": record.binding.tool_call_id,
                "tool_name": record.binding.tool_name,
            },
            scope=ExecutionScope(
                namespace=record.binding.namespace,
                thread_id=record.binding.thread_id,
                run_id=record.binding.run_id,
            ),
        )
        return record

    async def adecide_approval(
        self, request_id: str, *, approved: bool, decided_by: str, approvals=_UNSET
    ):
        return await asyncio.to_thread(
            self.decide_approval,
            request_id,
            approved=approved,
            decided_by=decided_by,
            approvals=approvals,
        )

    def _load_approval_snapshot(self, thread_id, *, approvals=_UNSET):
        store = self._get_effective_checkpoint_store()
        policy = self._get_effective_approvals(approvals)
        if store is None or policy is None:
            return ()
        state = store.load_latest_run(self.get_module_name(), thread_id)
        if state is None:
            return ()
        pending = state.get("runtime", {}).get("extensions", {}).get(_KEY, {})
        records = [
            policy.store.get(self.get_module_name(), request_id)
            for request_id in pending.get("requests", {}).values()
        ]
        return tuple(record for record in records if record is not None)

    def _approval_replay(self):
        run = get_agent_run()
        pending = run.get_extension(_KEY) if run is not None else None
        if pending is None:
            return None
        if pending.get("schema_version") != 1:
            raise TaskPauseRequestedError(message="Unsupported pending approval schema")
        if pending.get("phase") == "executing":
            raise ApprovalReconciliationRequiredError(
                message="Approval batch needs host reconciliation: no committed result"
            )
        if self._get_effective_approvals() is None:
            raise TaskPauseRequestedError(
                message="Pending approvals require the host approval configuration"
            )
        calls = ToolCallAggregator(api_mode=pending["api_mode"])
        calls.native_calls = deepcopy(pending.get("native_calls", {}))
        validate_native_calls(calls.native_calls, pending["intents"])
        for index, intent in enumerate(pending["intents"]):
            calls.process(
                index, intent["id"], intent["name"], msgspec_dumps(intent["arguments"])
            )
        response = ModelResponse()
        response.set_response_type("tool_call")
        response.add(calls)
        return response

    def _approval_pending(self, model_response, intents, messages):
        if self._get_effective_approvals() is None:
            return None
        run = get_agent_run()
        if run is None:
            raise ValueError("Use Agent.__call__ or Agent.acall for approval execution")
        pending = run.get_extension(_KEY)
        if pending is None and not any(
            intent.name in self._get_effective_approvals().tools for intent in intents
        ):
            return None
        store = self._get_effective_checkpoint_store()
        if not isinstance(messages, ChatMessages) or not getattr(
            store, "supports_atomic_commit", False
        ):
            raise ValueError(
                "Agent approvals require ChatMessages and atomic checkpoints"
            )
        if not isinstance(model_response.data, ToolCallAggregator):
            raise ValueError("Agent approvals require canonical tool-call responses")
        current = [
            {"id": intent.id, "name": intent.name, "arguments": dict(intent.arguments)}
            for intent in intents
        ]
        if pending is None:
            pending = {
                "schema_version": 1,
                "api_mode": model_response.data.api_mode,
                "native_calls": deepcopy(model_response.data.native_calls),
                "intents": current,
                "requests": {},
            }
            run.set_extension(_KEY, pending)
            pending = run.get_extension(_KEY)
        elif pending["intents"] != current:
            raise TaskPauseRequestedError(message="Pending approval batch changed")
        return pending

    def _process_approval_intents(
        self, model_response, intents, message, messages, vars
    ):
        pending = self._approval_pending(model_response, intents, messages)
        if pending is None:
            with (
                ApprovalBatch(
                    self._get_effective_approvals(), self.tool_library, {}
                ).activate()
                if self._get_effective_approvals() is not None
                else nullcontext()
            ):
                return self._process_tool_intents(intents, message, messages, vars)
        self._raise_if_background_task_interrupted()
        self._drain_inbox_into_messages(messages, vars=vars)
        batch = self._get_effective_approvals().prepare(
            self.tool_library, intents, pending
        )
        pending["requests"].update(
            {key: record.request_id for key, record in batch.records.items()}
        )
        self._checkpoint_save(messages, vars)
        batch.require_ready()
        pending["phase"] = "executing"
        self._checkpoint_save(messages, vars)
        with batch.activate():
            return self._process_tool_intents(intents, message, messages, vars)

    async def _aprocess_approval_intents(
        self, model_response, intents, message, messages, vars
    ):
        pending = self._approval_pending(model_response, intents, messages)
        if pending is None:
            with (
                ApprovalBatch(
                    self._get_effective_approvals(), self.tool_library, {}
                ).activate()
                if self._get_effective_approvals() is not None
                else nullcontext()
            ):
                return await self._aprocess_tool_intents(
                    intents, message, messages, vars
                )
        self._raise_if_background_task_interrupted()
        await self._adrain_inbox_into_messages(messages, vars=vars)
        batch = await asyncio.to_thread(
            self._get_effective_approvals().prepare, self.tool_library, intents, pending
        )
        pending["requests"].update(
            {key: record.request_id for key, record in batch.records.items()}
        )
        await self._acheckpoint_save(messages, vars)
        batch.require_ready()
        pending["phase"] = "executing"
        await self._acheckpoint_save(messages, vars)
        with batch.activate():
            return await self._aprocess_tool_intents(intents, message, messages, vars)

    @staticmethod
    def _clear_approval_batch():
        run = get_agent_run()
        if run is not None:
            run.extension_state.pop(_KEY, None)
