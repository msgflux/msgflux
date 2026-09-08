"""ToolLibrary preparation, policy, dispatch, and outcome pipeline."""

# ruff: noqa: A001, A002

import weakref
from dataclasses import replace
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import msgspec

import msgflux.nn.functional as F
from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import (
    AbortRequestedError,
    TaskError,
    TaskInterruptRequestedError,
)
from msgflux.nn.hooks.events import AfterTool, BeforeTool, BeforeToolDispatch
from msgflux.nn.modules.tool.implementations import Tool
from msgflux.nn.modules.tool.runtime import (
    AfterToolPolicy,
    BeforeDispatchPolicy,
    BeforeToolPolicy,
    DispatchRequest,
    ToolExecutionPlan,
    ToolRef,
    ToolRuntimeContext,
)
from msgflux.nn.modules.tool.runtime import (
    ToolDefinition as RuntimeToolDefinition,
)
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.events import EventType, emit_event, event_source
from msgflux.runtime.permissions import require_permissions
from msgflux.tools.helpers import (
    RESERVED_TOOL_KINDS,
    RUNTIME_BACKGROUND_PARAM,
    coerce_tool_params,
)
from msgflux.tools.responses import ToolResponses
from msgflux.tools.runtime import ToolIntent, ToolOutcome
from msgflux.tools.types import ToolBucket, ToolLibraryOperator


class _ToolBackgroundScheduler:
    """Adapt the durable task dispatcher to the canonical dispatch contract."""

    def __init__(self, library: Any) -> None:
        self._library_ref = weakref.ref(library)

    def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        library = self._library_ref()
        if library is None:
            raise RuntimeError("The ToolLibrary is no longer available")
        plan = request.plan
        call = library.get_background_dispatcher().dispatch(
            tool=plan.definition.executor,
            definition=plan.definition,
            tool_id=plan.intent.id,
            tool_name=plan.intent.name,
            call_params=plan.call_arguments,
            visible_params=plan.visible_arguments,
        )
        if call.error is not None:
            return library._failed_intent(
                plan.intent,
                status="execution_failed",
                code="tool_dispatch_failed",
                message=call.error,
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        return library._dispatched_intent(
            plan.intent,
            call.result,
            feedback=plan.feedback,
            arguments=plan.visible_arguments,
        )


class ToolLibraryExecutionMixin:
    """Internal execution pipeline mixed into the public ToolLibrary facade."""

    def _build_tool_argument_sets(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if definition.feedback.name == "handoff" or definition.metadata.get(
            "disable_input",
            False,
        ):
            visible_params: Dict[str, Any] = {}
        else:
            visible_params = coerce_tool_params(intent.name, tool_params)

        for param_name in definition.metadata.get("hidden_params") or {}:
            visible_params.pop(param_name, None)

        runtime_params: Dict[str, Any] = dict(runtime_arguments or {})
        if RUNTIME_BACKGROUND_PARAM in visible_params:
            runtime_params[RUNTIME_BACKGROUND_PARAM] = visible_params.pop(
                RUNTIME_BACKGROUND_PARAM
            )

        resolved = F.wait_for(
            self._aresolve_runtime_inputs,
            definition,
            intent,
            context,
        )
        if isinstance(resolved, TaskError):
            raise resolved.exception
        runtime_params.update(resolved)

        return visible_params, runtime_params

    async def _abuild_tool_argument_sets(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if definition.feedback.name == "handoff" or definition.metadata.get(
            "disable_input",
            False,
        ):
            visible_params: Dict[str, Any] = {}
        else:
            visible_params = coerce_tool_params(intent.name, tool_params)
        for param_name in definition.metadata.get("hidden_params") or {}:
            visible_params.pop(param_name, None)
        runtime_params: Dict[str, Any] = dict(runtime_arguments or {})
        if RUNTIME_BACKGROUND_PARAM in visible_params:
            runtime_params[RUNTIME_BACKGROUND_PARAM] = visible_params.pop(
                RUNTIME_BACKGROUND_PARAM
            )
        runtime_params.update(
            await self._aresolve_runtime_inputs(definition, intent, context)
        )
        return visible_params, runtime_params

    async def _aresolve_runtime_inputs(
        self,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        context: ToolRuntimeContext,
    ) -> Dict[str, Any]:
        try:
            return await await_with_abort(
                self.runtime_extensions.resolve_context(
                    definition,
                    intent,
                    context,
                ),
                context.get("abort_signal"),
            )
        except KeyError as exc:
            key = "unknown"
            for binding in definition.context.bindings:
                value = context.get(binding.source)
                selected_key = binding.options.get("key")
                if selected_key is not None and (
                    not isinstance(value, Mapping) or selected_key not in value
                ):
                    key = selected_key
                    break
                selected = binding.options.get("select") or ()
                missing = [
                    selected_key
                    for selected_key in selected
                    if not isinstance(value, Mapping) or selected_key not in value
                ]
                if missing:
                    key = missing[0]
                    break
            subject = "agent" if definition.kind == "agent" else "tool"
            raise ValueError(
                f"The {subject} `{intent.name}` requires the injected parameter "
                f"`{key}`, but it was not found."
            ) from exc

    def _record_tool_activity(
        self,
        *,
        activity_recorder: Any,
        definition: RuntimeToolDefinition,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        if (
            activity_recorder is None
            or definition.kind in RESERVED_TOOL_KINDS
            or ToolLibraryOperator.is_operator_tool(definition.executor)
        ):
            return
        activity_recorder.tool_call(definition.name, parameters)

    def _prepare_tool_kwargs(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        activity_recorder: Any,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        visible_params, runtime_params = self._build_tool_argument_sets(
            definition=definition,
            intent=intent,
            tool_params=tool_params,
            context=context,
            runtime_arguments=runtime_arguments,
        )
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            definition=definition,
            parameters=visible_params,
        )
        return visible_params, runtime_params

    async def _aprepare_tool_kwargs(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        tool_params: Any,
        context: ToolRuntimeContext,
        activity_recorder: Any,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        visible_params, runtime_params = await self._abuild_tool_argument_sets(
            definition=definition,
            intent=intent,
            tool_params=tool_params,
            context=context,
            runtime_arguments=runtime_arguments,
        )
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            definition=definition,
            parameters=visible_params,
        )
        return visible_params, runtime_params

    def _build_execution_plan(
        self,
        *,
        definition: RuntimeToolDefinition,
        intent: ToolIntent,
        visible_arguments: Mapping[str, Any],
        runtime_arguments: Mapping[str, Any],
    ) -> ToolExecutionPlan:
        selected_dispatch = definition.dispatch.name
        selected_runtime_arguments = dict(runtime_arguments)
        if selected_dispatch == "optional_background":
            selected_dispatch = (
                "background"
                if selected_runtime_arguments.pop(RUNTIME_BACKGROUND_PARAM, False)
                is True
                else "foreground"
            )
        if (
            selected_dispatch == "foreground"
            and definition.feedback.name != "call_as_response"
        ):
            selected_runtime_arguments["tool_call_id"] = intent.id
        return ToolExecutionPlan(
            intent=intent,
            definition=definition,
            visible_arguments=dict(visible_arguments),
            runtime_arguments=selected_runtime_arguments,
            dispatch=selected_dispatch,
        )

    @staticmethod
    def _with_dispatch_mode(
        plan: ToolExecutionPlan,
        dispatch_mode: str,
    ) -> ToolExecutionPlan:
        selected_dispatch = dispatch_mode
        if selected_dispatch == plan.dispatch.name:
            return plan
        runtime_arguments = dict(plan.runtime_arguments)
        runtime_arguments.pop("tool_call_id", None)
        runtime_arguments.pop(RUNTIME_BACKGROUND_PARAM, None)
        if (
            selected_dispatch == "foreground"
            and plan.feedback.name != "call_as_response"
        ):
            runtime_arguments["tool_call_id"] = plan.intent.id
        return ToolExecutionPlan(
            intent=plan.intent,
            definition=plan.definition,
            visible_arguments=plan.visible_arguments,
            runtime_arguments=runtime_arguments,
            dispatch=selected_dispatch,
            feedback=plan.feedback,
        )

    @staticmethod
    def _definition_config(
        definition: RuntimeToolDefinition,
    ) -> Mapping[str, Any]:
        return definition.declaration

    @classmethod
    def _plan_config(cls, plan: ToolExecutionPlan) -> Mapping[str, Any]:
        return cls._definition_config(plan.definition)

    def _tool_runtime_context(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        message: Any,
        messages: Any,
        vars: Mapping[str, Any],
        sync_dispatch: bool,
    ) -> ToolRuntimeContext:
        execution = get_execution_context()
        handle = self.get_handle().for_tool(
            tool_name=tool_name,
            agent_inbox=execution.get("agent_inbox"),
            task_store=execution.get("task_store"),
            message=message,
            messages=messages,
            vars=vars,
            tool_call_id=tool_call_id,
            activity_recorder=execution.get("task_activity_recorder"),
        )
        return ToolRuntimeContext(
            values={
                "message": message,
                "messages": messages,
                "vars": vars,
                "handle": handle,
                "abort_signal": execution.get("abort_signal"),
                "task_store": execution.get("task_store"),
                "agent_inbox": execution.get("agent_inbox"),
                "activity_recorder": execution.get("task_activity_recorder"),
                "background_dispatcher": _ToolBackgroundScheduler(self),
                "sync_dispatch": sync_dispatch,
            }
        )

    @staticmethod
    def _emit_tool_blocked(event: BeforeTool | BeforeToolDispatch) -> None:
        with event_source(event.tool_name, "tool"):
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": event.tool_call_id,
                    "tool_name": event.tool_name,
                    "arguments": dict(event.arguments),
                    "reason": event.block,
                },
            )

    @staticmethod
    def _emit_policy_blocked(
        intent: ToolIntent,
        outcome: ToolOutcome,
        arguments: Mapping[str, Any],
    ) -> None:
        reason = outcome.error.message if outcome.error is not None else "Tool blocked"
        with event_source(intent.name, "tool"):
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": intent.id,
                    "tool_name": intent.name,
                    "arguments": dict(arguments),
                    "reason": reason,
                },
            )

    @classmethod
    def _normalize_policy_outcome(
        cls,
        outcome: ToolOutcome,
        *,
        intent: ToolIntent,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        if outcome.intent_id != intent.id or outcome.tool_name != intent.name:
            raise ValueError("A policy returned an outcome for another tool intent")
        return msgspec.structs.replace(
            outcome,
            feedback=feedback,
            metadata={
                **dict(outcome.metadata),
                **cls._outcome_metadata(arguments),
            },
        )

    async def _abefore_tool_policy(
        self,
        intent: ToolIntent,
        definition: RuntimeToolDefinition,
        context: ToolRuntimeContext,
    ) -> BeforeToolPolicy | ToolOutcome:
        try:
            result = await await_with_abort(
                self.runtime_extensions.before_tool(
                    BeforeToolPolicy(
                        intent=intent,
                        definition=definition,
                        context=context,
                    )
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_tool policy failed closed: {exc}",
                feedback=definition.feedback,
            )
        if isinstance(result, ToolOutcome):
            return self._normalize_policy_outcome(
                result,
                intent=intent,
                feedback=definition.feedback,
                arguments=intent.arguments,
            )
        return result

    async def _abefore_dispatch_policy(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        try:
            result = await await_with_abort(
                self.runtime_extensions.before_dispatch(
                    BeforeDispatchPolicy(plan=plan, context=context)
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return self._failed_intent(
                plan.intent,
                status="blocked",
                code="tool_policy_failed",
                message=f"before_dispatch policy failed closed: {exc}",
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        if isinstance(result, ToolOutcome):
            return self._normalize_policy_outcome(
                result,
                intent=plan.intent,
                feedback=plan.feedback,
                arguments=plan.visible_arguments,
            )
        return result

    def _validate_before_dispatch_event(
        self,
        event: Any,
        initial_event: BeforeToolDispatch,
    ) -> BeforeToolDispatch:
        if not isinstance(event, BeforeToolDispatch):
            raise TypeError(
                "before_dispatch handlers must return BeforeToolDispatch or None"
            )
        protected_fields = ("tool_call_id", "tool_name", "arguments", "config")
        changed_fields = [
            name
            for name in protected_fields
            if getattr(event, name) != getattr(initial_event, name)
        ]
        if changed_fields:
            formatted = ", ".join(f"`{name}`" for name in changed_fields)
            raise ValueError(
                "before_dispatch handlers may only replace `dispatch_mode` or "
                f"`block`; changed protected fields: {formatted}"
            )
        try:
            self.runtime_extensions.get_dispatch(event.dispatch_mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported tool dispatch mode: `{event.dispatch_mode}`"
            ) from exc
        if event.dispatch_mode != initial_event.dispatch_mode and not (
            initial_event.dispatch_mode in {"background", "detached"}
            and event.dispatch_mode == "foreground"
        ):
            raise ValueError(
                "before_dispatch may only keep the selected mode or reduce "
                "`background`/`detached` dispatch to `foreground`"
            )
        return event

    @staticmethod
    def _validate_blocking_hook_payload(
        event_name: str,
        payload: Any,
        expected_type: type,
    ) -> Any:
        if not isinstance(payload, expected_type):
            raise TypeError(
                f"{event_name} handlers must return {expected_type.__name__} or None"
            )
        return payload

    def _run_owned_blocking_hooks(
        self,
        event_name: str,
        payload: Any,
        validator: Callable[[Any], Any],
    ) -> Any:
        def stop_when(current: Any) -> bool:
            return getattr(current, "block", None) is not None

        if self.has_lifecycle_hooks(event_name):
            payload = self._run_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        payload = validator(payload)
        owner = self._get_lifecycle_owner()
        if (
            payload.block is None
            and owner is not None
            and owner.has_lifecycle_hooks(event_name)
        ):
            payload = owner._run_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        return validator(payload)

    async def _arun_owned_blocking_hooks(
        self,
        event_name: str,
        payload: Any,
        validator: Callable[[Any], Any],
    ) -> Any:
        def stop_when(current: Any) -> bool:
            return getattr(current, "block", None) is not None

        if self.has_lifecycle_hooks(event_name):
            payload = await self._arun_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        payload = validator(payload)
        owner = self._get_lifecycle_owner()
        if (
            payload.block is None
            and owner is not None
            and owner.has_lifecycle_hooks(event_name)
        ):
            payload = await owner._arun_lifecycle_hooks(
                event_name,
                payload,
                stop_when=stop_when,
            )
        return validator(payload)

    def _run_before_dispatch_hook(
        self,
        plan: ToolExecutionPlan,
    ) -> BeforeToolDispatch:
        event = BeforeToolDispatch(
            tool_call_id=plan.intent.id,
            tool_name=plan.intent.name,
            arguments=plan.visible_arguments,
            config=self._plan_config(plan),
            dispatch_mode=plan.dispatch.name,
        )
        initial_event = event
        try:
            event = self._run_owned_blocking_hooks(
                "before_dispatch",
                event,
                partial(
                    self._validate_before_dispatch_event,
                    initial_event=initial_event,
                ),
            )
            return event
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return replace(
                initial_event,
                block=f"before_dispatch hook failed closed: {exc}",
            )

    async def _arun_before_dispatch_hook(
        self,
        plan: ToolExecutionPlan,
    ) -> BeforeToolDispatch:
        event = BeforeToolDispatch(
            tool_call_id=plan.intent.id,
            tool_name=plan.intent.name,
            arguments=plan.visible_arguments,
            config=self._plan_config(plan),
            dispatch_mode=plan.dispatch.name,
        )
        initial_event = event
        try:
            event = await self._arun_owned_blocking_hooks(
                "before_dispatch",
                event,
                partial(
                    self._validate_before_dispatch_event,
                    initial_event=initial_event,
                ),
            )
            return event
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return replace(
                initial_event,
                block=f"before_dispatch hook failed closed: {exc}",
            )

    def _run_before_tool_hook(
        self,
        *,
        tool_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> BeforeTool:
        event = BeforeTool(
            tool_call_id=tool_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        try:
            return self._run_owned_blocking_hooks(
                "before_tool",
                event,
                partial(
                    self._validate_blocking_hook_payload,
                    "before_tool",
                    expected_type=BeforeTool,
                ),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return BeforeTool(
                tool_call_id=tool_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                block=f"before_tool hook failed closed: {exc}",
            )

    async def _arun_before_tool_hook(
        self,
        *,
        tool_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> BeforeTool:
        event = BeforeTool(
            tool_call_id=tool_id,
            tool_name=tool_name,
            arguments=dict(arguments),
        )
        try:
            return await self._arun_owned_blocking_hooks(
                "before_tool",
                event,
                partial(
                    self._validate_blocking_hook_payload,
                    "before_tool",
                    expected_type=BeforeTool,
                ),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            return BeforeTool(
                tool_call_id=tool_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                block=f"before_tool hook failed closed: {exc}",
            )

    def _resolve_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
    ) -> Tool:
        if bucket_name not in self.library:
            raise ValueError(f"The bucket `{bucket_name}` is no longer available.")
        bucket = getattr(self.library[bucket_name], "impl", None)
        if not isinstance(bucket, ToolBucket):
            raise ValueError(f"The tool `{bucket_name}` is not a tool bucket.")
        if tool_name not in bucket.tools:
            available = ", ".join(sorted(bucket.tools)) or "none"
            raise ValueError(
                f"Tool `{tool_name}` is not captured by `{bucket_name}`. "
                f"Available tools: {available}."
            )
        definition = self.get_tool_definition(tool_name)
        executor = definition.executor
        if not isinstance(executor, Tool):
            raise TypeError(f"Tool `{tool_name}` has an invalid executor")
        return executor

    def _execute_prepared_tool(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> Any:
        event_data = self._tool_event_data(tool, call_params, visible_params)
        with event_source(event_data["tool_name"], "tool"):
            return self._execute_prepared_tool_impl(tool, call_params, event_data)

    def _execute_prepared_tool_impl(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        event_data: Mapping[str, Any],
    ) -> Any:
        emit_event(EventType.TOOL_START, event_data)
        try:
            abort_signal = get_execution_context().get("abort_signal")
            if abort_signal is not None:
                abort_signal.raise_if_aborted()
            result = tool(**call_params)
            if abort_signal is not None:
                abort_signal.raise_if_aborted()
        except BaseException as exc:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                error=exc,
            )
        else:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                result=result,
            )
        outcome = self._run_after_tool_hook(outcome)
        emit_event(
            EventType.TOOL_END,
            {
                **event_data,
                "result": outcome.result,
                "error": str(outcome.error) if outcome.error is not None else None,
            },
        )
        if outcome.error is not None:
            if isinstance(outcome.error, BaseException):
                raise outcome.error
            raise RuntimeError(str(outcome.error))
        return outcome.result

    async def _aexecute_prepared_tool(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> Any:
        event_data = self._tool_event_data(tool, call_params, visible_params)
        with event_source(event_data["tool_name"], "tool"):
            return await self._aexecute_prepared_tool_impl(
                tool,
                call_params,
                event_data,
            )

    async def _aexecute_prepared_tool_impl(
        self,
        tool: Tool,
        call_params: Mapping[str, Any],
        event_data: Mapping[str, Any],
    ) -> Any:
        emit_event(EventType.TOOL_START, event_data)
        try:
            result = await await_with_abort(
                tool.acall(**call_params),
                get_execution_context().get("abort_signal"),
            )
        except BaseException as exc:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                error=exc,
            )
        else:
            outcome = AfterTool(
                tool_call_id=event_data["tool_call_id"],
                tool_name=event_data["tool_name"],
                arguments=event_data["arguments"],
                result=result,
            )
        outcome = await self._arun_after_tool_hook(outcome)
        emit_event(
            EventType.TOOL_END,
            {
                **event_data,
                "result": outcome.result,
                "error": str(outcome.error) if outcome.error is not None else None,
            },
        )
        if outcome.error is not None:
            if isinstance(outcome.error, BaseException):
                raise outcome.error
            raise RuntimeError(str(outcome.error))
        return outcome.result

    def _run_after_tool_hook(self, outcome: AfterTool) -> AfterTool:
        try:
            if self.has_lifecycle_hooks("after_tool"):
                outcome = self._run_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("after_tool"):
                outcome = owner._run_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            return outcome
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            emit_event(
                EventType.HANDLER_ERROR,
                {"hook": "after_tool", "error": str(exc)},
            )
            return outcome

    async def _arun_after_tool_hook(self, outcome: AfterTool) -> AfterTool:
        try:
            if self.has_lifecycle_hooks("after_tool"):
                outcome = await self._arun_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            owner = self._get_lifecycle_owner()
            if owner is not None and owner.has_lifecycle_hooks("after_tool"):
                outcome = await owner._arun_lifecycle_hooks("after_tool", outcome)
            if not isinstance(outcome, AfterTool):
                raise TypeError("after_tool handlers must return AfterTool or None")
            return outcome
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception as exc:
            emit_event(
                EventType.HANDLER_ERROR,
                {"hook": "after_tool", "error": str(exc)},
            )
            return outcome

    @staticmethod
    def _tool_event_data(
        tool: Tool,
        call_params: Mapping[str, Any],
        visible_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "tool_call_id": call_params.get("tool_call_id"),
            "tool_name": tool.get_module_name(),
            "arguments": dict(visible_params),
        }

    def run(
        self,
        tool_ref: ToolRef | str,
        arguments: Mapping[str, Any],
        *,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Execute one logical tool reference through the library pipeline."""
        tool_name = self._resolve_tool_ref_name(tool_ref)
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        owner = bucket_name or self.name
        tool_call_id = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{owner}:{tool_name}"
        )
        intent = ToolIntent(id=tool_call_id, name=tool_name, arguments=arguments)
        context = self._tool_runtime_context(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            message=message,
            messages=messages,
            vars=vars,
            sync_dispatch=True,
        )
        recorder = (
            activity_recorder
            if activity_recorder is not None
            else get_execution_context().get("task_activity_recorder")
        )
        prepared = self._prepare_intent(
            intent=intent,
            context=context,
            messages=messages,
            activity_recorder=recorder,
            bucket_name=bucket_name,
            runtime_arguments=runtime_arguments,
        )
        if isinstance(prepared, ToolOutcome):
            return self._unwrap_handle_outcome(prepared)
        dispatched = self._dispatch_intent_plan(intent, prepared, context)
        outcome = dispatched if isinstance(dispatched, ToolOutcome) else dispatched()
        if isinstance(outcome, TaskError):
            raise outcome.exception
        return self._unwrap_handle_outcome(outcome)

    async def arun(
        self,
        tool_ref: ToolRef | str,
        arguments: Mapping[str, Any],
        *,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,
        parent_tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> Any:
        """Async counterpart of :meth:`run`."""
        tool_name = self._resolve_tool_ref_name(tool_ref)
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        owner = bucket_name or self.name
        tool_call_id = (
            f"{parent_tool_call_id}:{tool_name}"
            if parent_tool_call_id
            else f"{owner}:{tool_name}"
        )
        intent = ToolIntent(id=tool_call_id, name=tool_name, arguments=arguments)
        context = self._tool_runtime_context(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            message=message,
            messages=messages,
            vars=vars,
            sync_dispatch=False,
        )
        recorder = (
            activity_recorder
            if activity_recorder is not None
            else get_execution_context().get("task_activity_recorder")
        )
        prepared = await self._aprepare_intent(
            intent=intent,
            context=context,
            messages=messages,
            activity_recorder=recorder,
            bucket_name=bucket_name,
            runtime_arguments=runtime_arguments,
        )
        if isinstance(prepared, ToolOutcome):
            return self._unwrap_handle_outcome(prepared)
        dispatched = await self._adispatch_intent_plan(intent, prepared, context)
        outcome = (
            dispatched if isinstance(dispatched, ToolOutcome) else await dispatched()
        )
        return self._unwrap_handle_outcome(outcome)

    @staticmethod
    def _unwrap_handle_outcome(outcome: ToolOutcome) -> Any:
        """Preserve the handle's value-or-exception interface over outcomes."""
        if outcome.ok:
            return outcome.result
        if outcome.error is None:
            raise RuntimeError(
                f"Tool `{outcome.tool_name}` finished with status `{outcome.status}`."
            )
        if outcome.status == "not_found":
            raise ValueError(outcome.error.message)
        raise RuntimeError(outcome.error.message)

    def _resolve_tool_ref_name(self, tool_ref: ToolRef | str) -> str:
        if isinstance(tool_ref, ToolRef):
            if tool_ref.library_id != self.name:
                raise ValueError(
                    f"Tool ref belongs to `{tool_ref.library_id}`, not `{self.name}`"
                )
            return tool_ref.tool_id
        if not isinstance(tool_ref, str) or not tool_ref:
            raise TypeError("`tool_ref` must be a ToolRef or non-empty string")
        return tool_ref

    def _call_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Compatibility wrapper for the former bucket execution path."""
        return self.run(
            self.get_tool_ref(tool_name),
            arguments,
            bucket_name=bucket_name,
            **kwargs,
        )

    async def _acall_captured_tool(
        self,
        bucket_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Compatibility wrapper for the former async bucket execution path."""
        return await self.arun(
            self.get_tool_ref(tool_name),
            arguments,
            bucket_name=bucket_name,
            **kwargs,
        )

    def execute_intents(
        self,
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
        *,
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ToolOutcome, ...]:
        """Execute canonical intents without lowering them to legacy responses."""
        normalized = self._validate_intents(intents)
        return self._execute_intent_batch(
            normalized,
            message=message,
            messages=messages,
            vars=vars,
        )

    async def aexecute_intents(
        self,
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
        *,
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[ToolOutcome, ...]:
        """Async counterpart of execute_intents."""
        normalized = self._validate_intents(intents)
        return await self._aexecute_intent_batch(
            normalized,
            message=message,
            messages=messages,
            vars=vars,
        )

    @staticmethod
    def _validate_intents(
        intents: List[ToolIntent] | Tuple[ToolIntent, ...],
    ) -> Tuple[ToolIntent, ...]:
        normalized = tuple(intents)
        if not all(isinstance(intent, ToolIntent) for intent in normalized):
            raise TypeError("`intents` must contain ToolIntent values")
        return normalized

    @staticmethod
    def _outcome_metadata(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        return {"arguments": dict(arguments)}

    @classmethod
    def _failed_intent(
        cls,
        intent: ToolIntent,
        *,
        status: str,
        code: str,
        message: str,
        feedback: Any = None,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        return ToolOutcome.failed(
            intent,
            status=status,
            code=code,
            message=message,
            feedback=feedback,
            metadata=cls._outcome_metadata(
                intent.arguments if arguments is None else arguments
            ),
        )

    @classmethod
    def _completed_intent(
        cls,
        intent: ToolIntent,
        result: Any,
        *,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        return ToolOutcome.completed(
            intent,
            result,
            feedback=feedback,
            metadata=cls._outcome_metadata(arguments),
        )

    @classmethod
    def _dispatched_intent(
        cls,
        intent: ToolIntent,
        result: Any,
        *,
        feedback: Any,
        arguments: Mapping[str, Any],
    ) -> ToolOutcome:
        return ToolOutcome.dispatched(
            intent,
            result,
            feedback=feedback,
            metadata=cls._outcome_metadata(arguments),
        )

    def _prepare_intent(
        self,
        intent: ToolIntent,
        *,
        context: ToolRuntimeContext,
        messages: ChatMessages | List[Dict[str, Any]],
        activity_recorder: Any,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionPlan | ToolOutcome:
        resolved = (
            self._resolve_captured_tool(bucket_name, intent.name)
            if bucket_name is not None
            else self._resolve_tool(intent.name)
        )
        if resolved is None:
            return self._failed_intent(
                intent,
                status="not_found",
                code="tool_not_found",
                message=f"Error: Tool `{intent.name}` not found.",
            )

        definition = self.get_tool_definition(intent.name)
        denied = self._permission_outcome(intent, definition)
        if denied is not None:
            return denied
        before_policy = F.wait_for(
            self._abefore_tool_policy,
            intent,
            definition,
            context,
        )
        if isinstance(before_policy, ToolOutcome):
            self._emit_policy_blocked(intent, before_policy, intent.arguments)
            return before_policy
        intent = before_policy.intent
        if definition.loading.deferred and isinstance(messages, ChatMessages):
            messages.load_tools(self.name, [intent.name])
        visible_arguments, runtime_arguments = self._prepare_tool_kwargs(
            definition=definition,
            intent=intent,
            tool_params=intent.arguments,
            context=context,
            activity_recorder=activity_recorder,
            runtime_arguments=runtime_arguments,
        )
        feedback = definition.feedback
        before_tool = self._run_before_tool_hook(
            tool_id=intent.id,
            tool_name=intent.name,
            arguments=visible_arguments,
        )
        if before_tool.block is not None:
            self._emit_tool_blocked(before_tool)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_blocked",
                message=before_tool.block,
                feedback=feedback,
                arguments=before_tool.arguments,
            )
        response_arguments = dict(before_tool.arguments)
        plan = self._build_execution_plan(
            definition=definition,
            intent=intent,
            visible_arguments=response_arguments,
            runtime_arguments=runtime_arguments,
        )
        before_dispatch = self._run_before_dispatch_hook(plan)
        if before_dispatch.block is not None:
            self._emit_tool_blocked(before_dispatch)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_dispatch_blocked",
                message=before_dispatch.block,
                feedback=feedback,
                arguments=response_arguments,
            )
        plan = self._with_dispatch_mode(plan, before_dispatch.dispatch_mode)
        before_policy_dispatch = F.wait_for(
            self._abefore_dispatch_policy,
            plan,
            context,
        )
        if isinstance(before_policy_dispatch, ToolOutcome):
            self._emit_policy_blocked(
                intent,
                before_policy_dispatch,
                response_arguments,
            )
            return before_policy_dispatch
        return before_policy_dispatch.plan

    async def _aprepare_intent(
        self,
        intent: ToolIntent,
        *,
        context: ToolRuntimeContext,
        messages: ChatMessages | List[Dict[str, Any]],
        activity_recorder: Any,
        bucket_name: str | None = None,
        runtime_arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionPlan | ToolOutcome:
        resolved = (
            self._resolve_captured_tool(bucket_name, intent.name)
            if bucket_name is not None
            else self._resolve_tool(intent.name)
        )
        if resolved is None:
            return self._failed_intent(
                intent,
                status="not_found",
                code="tool_not_found",
                message=f"Error: Tool `{intent.name}` not found.",
            )

        definition = self.get_tool_definition(intent.name)
        denied = self._permission_outcome(intent, definition)
        if denied is not None:
            return denied
        before_policy = await self._abefore_tool_policy(
            intent,
            definition,
            context,
        )
        if isinstance(before_policy, ToolOutcome):
            self._emit_policy_blocked(intent, before_policy, intent.arguments)
            return before_policy
        intent = before_policy.intent
        if definition.loading.deferred and isinstance(messages, ChatMessages):
            messages.load_tools(self.name, [intent.name])
        visible_arguments, runtime_arguments = await self._aprepare_tool_kwargs(
            definition=definition,
            intent=intent,
            tool_params=intent.arguments,
            context=context,
            activity_recorder=activity_recorder,
            runtime_arguments=runtime_arguments,
        )
        feedback = definition.feedback
        before_tool = await self._arun_before_tool_hook(
            tool_id=intent.id,
            tool_name=intent.name,
            arguments=visible_arguments,
        )
        if before_tool.block is not None:
            self._emit_tool_blocked(before_tool)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_blocked",
                message=before_tool.block,
                feedback=feedback,
                arguments=before_tool.arguments,
            )
        response_arguments = dict(before_tool.arguments)
        plan = self._build_execution_plan(
            definition=definition,
            intent=intent,
            visible_arguments=response_arguments,
            runtime_arguments=runtime_arguments,
        )
        before_dispatch = await self._arun_before_dispatch_hook(plan)
        if before_dispatch.block is not None:
            self._emit_tool_blocked(before_dispatch)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_dispatch_blocked",
                message=before_dispatch.block,
                feedback=feedback,
                arguments=response_arguments,
            )
        plan = self._with_dispatch_mode(plan, before_dispatch.dispatch_mode)
        before_policy_dispatch = await self._abefore_dispatch_policy(plan, context)
        if isinstance(before_policy_dispatch, ToolOutcome):
            self._emit_policy_blocked(
                intent,
                before_policy_dispatch,
                response_arguments,
            )
            return before_policy_dispatch
        return before_policy_dispatch.plan

    def _dispatch_intent_plan(
        self,
        intent: ToolIntent,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome | Callable[[], Any]:
        denied = self._permission_outcome(plan.intent, plan.definition)
        if denied is not None:
            return denied
        feedback = plan.feedback
        arguments = plan.visible_arguments
        if feedback.name == "call_as_response":
            return self._completed_intent(
                intent,
                None,
                feedback=feedback,
                arguments=arguments,
            )
        return partial(
            F.wait_for,
            self._adispatch_runtime_plan,
            plan,
            context,
        )

    async def _adispatch_intent_plan(
        self,
        intent: ToolIntent,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome | Callable[[], Any]:
        denied = self._permission_outcome(plan.intent, plan.definition)
        if denied is not None:
            return denied
        feedback = plan.feedback
        arguments = plan.visible_arguments
        if feedback.name == "call_as_response":
            return self._completed_intent(
                intent,
                None,
                feedback=feedback,
                arguments=arguments,
            )
        return partial(self._adispatch_runtime_plan, plan, context)

    def _permission_outcome(
        self, intent: ToolIntent, definition: RuntimeToolDefinition
    ) -> ToolOutcome | None:
        try:
            require_permissions(definition.required_permissions)
            # A transformed plan cannot drop its registered requirements.
            require_permissions(
                self.get_tool_definition(intent.name).required_permissions
            )
        except PermissionError as exc:
            payload = {
                "tool_call_id": intent.id,
                "tool_name": intent.name,
                "code": "tool_permission_denied",
            }
            emit_event(EventType.TOOL_PERMISSION_DENIED, payload)
            emit_event(EventType.TOOL_BLOCKED, payload)
            return self._failed_intent(
                intent,
                status="blocked",
                code="tool_permission_denied",
                message=str(exc),
                arguments={},
            )
        return None

    async def _adispatch_runtime_plan(
        self,
        plan: ToolExecutionPlan,
        context: ToolRuntimeContext,
    ) -> ToolOutcome:
        denied = self._permission_outcome(plan.intent, plan.definition)
        if denied is not None:
            return denied
        execution_denial = None

        async def execute(
            selected_plan: ToolExecutionPlan | None = None,
        ) -> ToolOutcome:
            nonlocal execution_denial
            if execution_denial is not None:
                return execution_denial
            current = selected_plan or plan
            denied = self._permission_outcome(current.intent, current.definition)
            if denied is not None:
                execution_denial = denied
                return denied
            result = await self._aexecute_prepared_tool(
                current.definition.executor,
                current.call_arguments,
                current.visible_arguments,
            )
            return self._completed_intent(
                current.intent,
                result,
                feedback=current.feedback,
                arguments=current.visible_arguments,
            )

        outcome = await await_with_abort(
            self.runtime_extensions.dispatch(
                DispatchRequest(plan=plan, context=context, execute=execute)
            ),
            context.get("abort_signal"),
        )
        if execution_denial is not None:
            return execution_denial
        result = outcome.result
        if plan.dispatch.name == "detached" and result is None:
            result = (
                f"The `{plan.intent.name}` tool was dispatched. "
                "This tool will not generate a return."
            )
        outcome = ToolOutcome(
            intent_id=outcome.intent_id,
            tool_name=outcome.tool_name,
            status=outcome.status,
            result=result,
            error=outcome.error,
            feedback=plan.feedback,
            metadata={
                **dict(outcome.metadata),
                **self._outcome_metadata(plan.visible_arguments),
            },
        )
        if outcome.intent_id != plan.intent.id or outcome.tool_name != plan.intent.name:
            raise ValueError("Dispatch returned an outcome for another tool intent")
        try:
            after_policy = await await_with_abort(
                self.runtime_extensions.after_tool(
                    AfterToolPolicy(
                        plan=plan,
                        outcome=outcome,
                        context=context,
                    )
                ),
                context.get("abort_signal"),
            )
        except (AbortRequestedError, TaskInterruptRequestedError):
            raise
        except Exception:
            return outcome
        return self._normalize_policy_outcome(
            after_policy.outcome,
            intent=plan.intent,
            feedback=plan.feedback,
            arguments=plan.visible_arguments,
        )

    def _execute_intent_batch(
        self,
        intents: Tuple[ToolIntent, ...],
        *,
        message: Any,
        messages: ChatMessages | List[Dict[str, Any]] | None,
        vars: Mapping[str, Any] | None,
    ) -> Tuple[ToolOutcome, ...]:
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        activity_recorder = get_execution_context().get("task_activity_recorder")
        outcomes: List[ToolOutcome | None] = [None] * len(intents)
        prepared_calls = []
        prepared_metadata = []

        for index, intent in enumerate(intents):
            runtime_context = self._tool_runtime_context(
                tool_name=intent.name,
                tool_call_id=intent.id,
                message=message,
                messages=messages,
                vars=vars,
                sync_dispatch=True,
            )
            prepared = self._prepare_intent(
                intent,
                context=runtime_context,
                messages=messages,
                activity_recorder=activity_recorder,
            )
            if isinstance(prepared, ToolOutcome):
                outcomes[index] = prepared
                continue
            dispatched = self._dispatch_intent_plan(
                intent,
                prepared,
                runtime_context,
            )
            if isinstance(dispatched, ToolOutcome):
                outcomes[index] = dispatched
                continue
            prepared_calls.append(dispatched)
            prepared_metadata.append((index, intent, prepared))

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for (index, intent, plan), result in zip(prepared_metadata, results):
                if isinstance(result, ToolOutcome):
                    outcomes[index] = result
                elif isinstance(result, TaskError):
                    if isinstance(
                        result.exception,
                        (AbortRequestedError, TaskInterruptRequestedError),
                    ):
                        raise result.exception
                    outcomes[index] = self._failed_intent(
                        intent,
                        status="execution_failed",
                        code="tool_execution_failed",
                        message=str(result),
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
                else:
                    outcomes[index] = self._completed_intent(
                        intent,
                        result,
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
        return self._finalize_outcomes(outcomes)

    async def _aexecute_intent_batch(
        self,
        intents: Tuple[ToolIntent, ...],
        *,
        message: Any,
        messages: ChatMessages | List[Dict[str, Any]] | None,
        vars: Mapping[str, Any] | None,
    ) -> Tuple[ToolOutcome, ...]:
        messages = messages if messages is not None else ChatMessages()
        vars = vars if vars is not None else {}
        activity_recorder = get_execution_context().get("task_activity_recorder")
        outcomes: List[ToolOutcome | None] = [None] * len(intents)
        prepared_calls = []
        prepared_metadata = []

        for index, intent in enumerate(intents):
            runtime_context = self._tool_runtime_context(
                tool_name=intent.name,
                tool_call_id=intent.id,
                message=message,
                messages=messages,
                vars=vars,
                sync_dispatch=False,
            )
            prepared = await self._aprepare_intent(
                intent,
                context=runtime_context,
                messages=messages,
                activity_recorder=activity_recorder,
            )
            if isinstance(prepared, ToolOutcome):
                outcomes[index] = prepared
                continue
            dispatched = await self._adispatch_intent_plan(
                intent,
                prepared,
                runtime_context,
            )
            if isinstance(dispatched, ToolOutcome):
                outcomes[index] = dispatched
                continue
            prepared_calls.append(dispatched)
            prepared_metadata.append((index, intent, prepared))

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for (index, intent, plan), result in zip(prepared_metadata, results):
                if isinstance(result, ToolOutcome):
                    outcomes[index] = result
                elif isinstance(result, TaskError):
                    if isinstance(
                        result.exception,
                        (AbortRequestedError, TaskInterruptRequestedError),
                    ):
                        raise result.exception
                    outcomes[index] = self._failed_intent(
                        intent,
                        status="execution_failed",
                        code="tool_execution_failed",
                        message=str(result),
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
                else:
                    outcomes[index] = self._completed_intent(
                        intent,
                        result,
                        feedback=self.get_tool_definition(intent.name).feedback,
                        arguments=plan.visible_arguments,
                    )
        return self._finalize_outcomes(outcomes)

    @staticmethod
    def _finalize_outcomes(
        outcomes: List[ToolOutcome | None],
    ) -> Tuple[ToolOutcome, ...]:
        missing = [index for index, outcome in enumerate(outcomes) if outcome is None]
        if missing:
            formatted = ", ".join(str(index) for index in missing)
            raise RuntimeError(f"Tool outcomes are missing at indexes: {formatted}")
        return tuple(outcome for outcome in outcomes if outcome is not None)

    @staticmethod
    def _outcomes_to_responses(
        intents: Tuple[ToolIntent, ...],
        outcomes: Tuple[ToolOutcome, ...],
    ) -> ToolResponses:
        return ToolResponses.from_outcomes(intents, outcomes)

    @staticmethod
    def _legacy_calls_to_intents(
        tool_callings: List[Tuple[str, str, Any]],
    ) -> Tuple[ToolIntent, ...]:
        return tuple(
            ToolIntent(
                id=tool_id,
                name=tool_name,
                arguments=coerce_tool_params(tool_name, tool_params),
            )
            for tool_id, tool_name, tool_params in tool_callings
        )

    def forward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Executes tool calls with tool config logic.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', {})]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        intents = self._legacy_calls_to_intents(tool_callings)
        outcomes = self.execute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        return self._outcomes_to_responses(intents, outcomes)

    async def aforward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Async version of forward. Executes tool calls with logic for
        `handoff`, `return_direct`.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', {})]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        intents = self._legacy_calls_to_intents(tool_callings)
        outcomes = await self.aexecute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        return self._outcomes_to_responses(intents, outcomes)
