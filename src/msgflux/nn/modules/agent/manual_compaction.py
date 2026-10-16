"""Scope-only checkpoint maintenance, independent of automatic run settlement."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy

import msgspec

from msgflux._private.response_metadata import minimal_usage_metadata
from msgflux.chat_messages import ChatMessages
from msgflux.nn.hooks.events import BeforeCompaction
from msgflux.nn.modules.agent.context import _agent_context, _require_lifecycle_payload
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.compaction import (
    CompactionBusyError,
    CompactionRecoveryRequiredError,
    CompactionResult,
)
from msgflux.runtime.context import (
    DEFAULT_NAMESPACE,
    ExecutionScope,
    execution_context,
)
from msgflux.runtime.events import (
    EventType,
    _capture_events,
    _hub_event_sink,
    _is_capturing_events,
    emit_event,
    event_source,
)
from msgflux.runtime.workspace.receipts import (
    task_command_receipts,
    unresolved_command_receipts,
)


class _Operation(msgspec.Struct):
    store: object
    state: dict
    scope: ExecutionScope
    messages: ChatMessages
    boundary: str | None

    def result(self, reason, *, operation=None, usage=None):
        return CompactionResult(
            self.scope.thread_id,
            self.scope.run_id,
            "completed" if operation is not None else "skipped",
            reason,
            operation_id=operation.get("item_id") if operation is not None else None,
            usage=usage,
        )


@contextmanager
def _operation(agent, scope):
    if not isinstance(scope, ExecutionScope):
        raise TypeError("scope must be an ExecutionScope")
    if not isinstance(scope.thread_id, str) or not scope.thread_id.strip():
        raise ValueError("Manual compaction requires an explicit thread_id")
    if scope.namespace not in (None, DEFAULT_NAMESPACE, agent.get_module_name()):
        raise ValueError("Compaction scope namespace does not match the Agent")
    scope = agent._workspace_scope(scope).with_overrides(
        namespace=agent.get_module_name()
    )
    if scope.abort_signal is not None:
        scope.abort_signal.raise_if_aborted()
    with agent._thread_operation(scope.thread_id, exclusive=True):
        if agent._get_effective_checkpoint_store(scope.thread_id) is None:
            raise ValueError("Manual compaction requires an existing checkpoint store")
        kwargs = {"scope": scope}
        with agent._bound_resource_context(kwargs):
            store = agent._get_effective_checkpoint_store()
            if store is None or not getattr(store, "supports_atomic_commit", False):
                raise ValueError(
                    "Manual compaction requires an atomic checkpoint store"
                )
            state = store.load_latest_run(agent.get_module_name(), scope.thread_id)
            if state is None:
                raise KeyError(f"No checkpoint for thread {scope.thread_id}")
            state = deepcopy(state)
            run_id = _saved_run_id(state, scope)
            scope = scope.with_overrides(
                run_id=run_id,
                account_name=scope.account_name
                or state.get("scope", {}).get("account_name"),
                principal=scope.principal or state.get("scope", {}).get("principal"),
            )
            with (
                execution_context(scope=scope),
                nullcontext()
                if _is_capturing_events()
                else _capture_events(_hub_event_sink(root_module=agent)),
                event_source(agent.get_module_name(), "agent"),
                agent._extension_run_snapshot(),
                _agent_context(agent, scope=scope, vars={}),
            ):
                _validate_saved_work(agent, state, scope)
                agent._restore_agent_run(state, scope.thread_id, run_id)
                messages = ChatMessages()
                messages._hydrate_state(state.get("messages", {}))
                boundary = messages.latest_completed_turn_boundary()
                yield _Operation(
                    store,
                    state,
                    scope,
                    messages,
                    boundary["item_id"] if boundary else None,
                )


def _saved_run_id(state, scope):
    run_id = state.get("scope", {}).get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CompactionRecoveryRequiredError("Checkpoint has no run identity")
    if scope.run_id is not None and scope.run_id != run_id:
        raise ValueError("Only the latest saved run can be compacted")
    if state.get("status") in {"running", "streaming"}:
        raise CompactionBusyError("The saved run is still active")
    if state.get("status") not in {"paused", "failed"}:
        raise ValueError("Manual compaction requires a paused or failed run")
    return run_id


def _validate_saved_work(agent, state, scope):
    try:
        agent._validate_checkpoint_command_receipts(state)
        agent._validate_checkpoint_workspace(state, scope=scope)
        reason = agent._uncertain_checkpoint_approval_reason(state)
        reason = reason or agent._active_background_future_reason((agent.tool_library,))
        store = agent.tool_library.get_task_store()
        if not reason and store is not None:
            reason = agent._unfinished_task_reason(scope.thread_id, store)
            if not reason:
                for task in store.list_summaries(
                    thread_id=scope.thread_id, run_id=scope.run_id
                ):
                    if unresolved_command_receipts(
                        task_command_receipts(store, task.task_id)
                    ):
                        reason = "A background command requires reconciliation"
                        break
        if reason:
            raise CompactionRecoveryRequiredError(reason)
    except CompactionRecoveryRequiredError:
        raise
    except (RuntimeError, ValueError, OSError) as error:
        raise CompactionRecoveryRequiredError(str(error)) from error


def _skip(agent, operation):
    if operation.boundary is None:
        return operation.result("no_completed_turn")
    if agent._boundary_is_already_compacted(
        operation.messages, {"item_id": operation.boundary}
    ):
        return operation.result("already_compacted")
    return None


def _parameters(agent, operation, *, asynchronous=False):
    return agent._prepare_model_execution(
        operation.messages,
        {},
        scope=operation.scope,
        model_messages=operation.messages.copy(),
        model_preference=operation.state.get("model_preference"),
        drain_notifications=False,
        transform_tool_catalog=not asynchronous,
        transform_system_prompt=not asynchronous,
    )


def _decision(agent, operation, estimate):
    return BeforeCompaction(
        scope=operation.scope,
        vars={},
        messages=operation.messages.copy(),
        estimated_input_tokens=estimate.input_tokens,
        estimate_source=estimate.source,
        context_capacity=getattr(agent.model, "context_capacity", None),
        compacted_through_item_id=operation.boundary,
        reason="manual",
        action="compact",
    )


def _validate_decision(operation, decision):
    decision = _require_lifecycle_payload(
        "before_compaction", decision, BeforeCompaction
    )
    if (
        decision.compacted_through_item_id != operation.boundary
        or decision.scope != operation.scope
    ):
        raise ValueError(
            "Compaction hooks cannot change the saved run or completed-turn boundary"
        )
    return decision


def _apply(agent, operation, decision, compacted):
    usage = minimal_usage_metadata({"usage": compacted.usage})
    item = operation.messages.add_compaction(
        compacted_through_item_id=operation.boundary,
        views=[compacted.to_view()],
        reason="manual",
        metadata=agent._compaction_metadata(compacted, decision, usage),
    )
    state = deepcopy(operation.state)
    state["messages"] = operation.messages._to_state()
    params = {
        "expected_revision": operation.state.get("_checkpoint", {}).get("revision", 0),
        "event": {
            "event_type": "compaction",
            "status": state["status"],
            "operation_id": item["item_id"],
        },
    }
    return state, params, item, usage


def _end(agent, operation, decision, *, item=None, usage=None, error=None):
    data = {
        **agent._compaction_event_data(decision),
        "status": "failed" if error is not None else "completed",
    }
    if error is not None:
        data["error"] = str(error)
    else:
        data.update(operation_id=item["item_id"], usage=usage)
    emit_event(EventType.COMPACTION_END, data, scope=operation.scope)


def compact(agent, scope):
    with _operation(agent, scope) as operation:
        skipped = _skip(agent, operation)
        if skipped is not None:
            return skipped
        params = _parameters(agent, operation)
        preference = operation.state.get("model_preference")
        estimate = agent._count_context_tokens(
            agent.model,
            params.messages,
            system_prompt=params.system_prompt,
            tool_catalog=params.tool_catalog,
            model_preference=preference,
        )
        decision = _validate_decision(
            operation,
            agent._run_lifecycle_hooks(
                "before_compaction", _decision(agent, operation, estimate)
            ),
        )
        if decision.action != "compact":
            return operation.result("policy_skipped")
        emit_event(
            EventType.COMPACTION_START,
            agent._compaction_event_data(decision),
            scope=operation.scope,
        )
        try:
            if operation.scope.abort_signal is not None:
                operation.scope.abort_signal.raise_if_aborted()
            view = agent._compact_context(
                agent.model,
                operation.messages.through_item(operation.boundary),
                system_prompt=params.system_prompt,
                model_preference=preference,
            )
            state, commit, item, usage = _apply(agent, operation, decision, view)
            if operation.scope.abort_signal is not None:
                operation.scope.abort_signal.raise_if_aborted()
            operation.store.commit_state(
                agent.get_module_name(),
                scope.thread_id,
                operation.scope.run_id,
                state,
                **commit,
            )
        except BaseException as error:
            _end(agent, operation, decision, error=error)
            raise
        _end(agent, operation, decision, item=item, usage=usage)
        return operation.result("manual", operation=item, usage=usage)


async def acompact(agent, scope):
    with _operation(agent, scope) as operation:
        skipped = _skip(agent, operation)
        if skipped is not None:
            return skipped
        params = _parameters(agent, operation, asynchronous=True)
        params = await agent._atransform_model_tool_catalog(
            params, vars={}, scope=operation.scope
        )
        params.system_prompt = agent._build_system_prompt(
            vars={}, tool_catalog=params.tool_catalog, apply_hooks=False
        )
        params = await agent._atransform_model_system_prompt(
            params, vars={}, scope=operation.scope
        )
        preference = operation.state.get("model_preference")
        estimate = await await_with_abort(
            agent._acount_context_tokens(
                agent.model,
                params.messages,
                system_prompt=params.system_prompt,
                tool_catalog=params.tool_catalog,
                model_preference=preference,
            ),
            operation.scope.abort_signal,
        )
        decision = _validate_decision(
            operation,
            await agent._arun_lifecycle_hooks(
                "before_compaction", _decision(agent, operation, estimate)
            ),
        )
        if decision.action != "compact":
            return operation.result("policy_skipped")
        emit_event(
            EventType.COMPACTION_START,
            agent._compaction_event_data(decision),
            scope=operation.scope,
        )
        try:
            view = await await_with_abort(
                agent._acompact_context(
                    agent.model,
                    operation.messages.through_item(operation.boundary),
                    system_prompt=params.system_prompt,
                    model_preference=preference,
                ),
                operation.scope.abort_signal,
            )
            state, commit, item, usage = _apply(agent, operation, decision, view)
            if operation.scope.abort_signal is not None:
                operation.scope.abort_signal.raise_if_aborted()
            # Keep the CAS synchronous: cancellation cannot detach a pending SQLite
            # write and release the maintenance fence before its outcome is known.
            operation.store.commit_state(
                agent.get_module_name(),
                scope.thread_id,
                operation.scope.run_id,
                state,
                **commit,
            )
        except BaseException as error:
            _end(agent, operation, decision, error=error)
            raise
        _end(agent, operation, decision, item=item, usage=usage)
        return operation.result("manual", operation=item, usage=usage)
