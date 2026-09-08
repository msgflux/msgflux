# ruff: noqa: A002

from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from msgflux._private.response_metadata import attach_response_metadata
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.core.message import Message
from msgflux.exceptions import (
    TaskInterruptRequestedError,
)
from msgflux.generation.control_flow import ToolFlowControl
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.events import emit_model_response_events
from msgflux.nn.functional import adetached, await_for_event, detached, wait_for_event
from msgflux.nn.hooks.events import (
    ConversationContext,
    ModelContext,
    ModelRequestContext,
    ModelResponseContext,
    ToolCatalogContext,
    ToolFeedbackContext,
)
from msgflux.nn.modules.tool import ToolResponses
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
)
from msgflux.runtime.context_scopes import ContextScopeController
from msgflux.runtime.events import EventType, emit_event
from msgflux.tools.catalog import ToolCatalogView
from msgflux.tools.helpers import coerce_tool_params
from msgflux.tools.runtime import ToolIntent, ToolOutcome
from msgflux.utils.console import cprint
from msgflux.utils.validation import is_subclass_of

if TYPE_CHECKING:
    pass
from msgflux.nn.modules.agent.context import (
    ToolFilter,
    _BeforeRunEndHookError,
    _require_lifecycle_payload,
)
from msgflux.nn.modules.agent.continuation import _TerminalResponse


class AgentModelRuntimeMixin:
    """Model request preparation, response processing, and tool-loop behavior."""

    _context_scope_tool_kind = "context_scope"

    def _is_context_scope_tool(self, tool_name: Any) -> bool:
        """Return whether a registered tool owns context-scope transitions.

        The result payload is deliberately not inspected here: arbitrary tools
        are allowed to return mappings with a ``type`` field, and those must not
        gain the ability to mutate conversation history.
        """
        if not isinstance(tool_name, str):
            return False
        try:
            definition = self.tool_library.get_tool_definition(tool_name)
        except (KeyError, ValueError):
            return False
        return definition.kind == self._context_scope_tool_kind

    def _validate_context_scope_intents(self, intents: Any) -> None:
        """Reject mixed scope transitions before any tool in the batch runs."""
        normalized_intents = tuple(intents)
        scope_intents = tuple(
            intent
            for intent in normalized_intents
            if self._is_context_scope_tool(getattr(intent, "name", None))
        )
        if scope_intents and len(normalized_intents) != 1:
            names = ", ".join(intent.name for intent in scope_intents)
            raise ValueError(
                "Context scope transitions are exclusive tool calls; "
                f"cannot execute `{names}` with other tools in one batch."
            )

    @staticmethod
    def _scope_command_value(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        if value.get("type") != "context_scope_transition":
            return None
        return value

    def _apply_context_scope_outcomes(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        outcomes: Any,
    ) -> None:
        if not isinstance(messages, ChatMessages):
            return
        commands = []
        for outcome in outcomes:
            if getattr(outcome, "status", None) != "completed":
                continue
            if not self._is_context_scope_tool(getattr(outcome, "tool_name", None)):
                continue
            command = self._scope_command_value(getattr(outcome, "result", None))
            if command is not None:
                commands.append(command)
        if len(commands) > 1:
            raise ValueError("Only one context scope transition may settle per batch")
        if commands:
            ContextScopeController().apply_command(messages, commands[0])

    def _execute_model(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        scope: Optional[ExecutionScope] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        self._raise_if_background_task_interrupted()
        decision = self._resolve_continuation("before_request", messages, vars)
        if decision.action == "return":
            return _TerminalResponse(decision)
        effective_scope = scope or get_execution_context()["scope"]
        conversation = self._run_lifecycle_hooks(
            "transform_context",
            ConversationContext(
                scope=effective_scope,
                vars=vars,
                messages=messages,
            ),
        )
        conversation = _require_lifecycle_payload(
            "transform_context", conversation, ConversationContext
        )
        model_execution_params = self._prepare_model_execution(
            messages=conversation.messages,
            prefilling=prefilling,
            model_preference=model_preference,
            vars=vars,
            tool_filter=tool_filter,
            scope=effective_scope,
        )
        compacted = self._maybe_compact_model_context(
            conversation.messages,
            model_execution_params.messages,
            system_prompt=model_execution_params.system_prompt,
            tool_catalog=model_execution_params.tool_catalog,
            model_preference=model_preference,
            vars=vars,
            scope=effective_scope,
        )
        if compacted:
            model_execution_params.messages = self._build_model_messages(
                conversation.messages,
                vars=vars,
                scope=effective_scope,
                drain_notifications=False,
            )
        request = ModelRequestContext.from_parameters(
            model_execution_params,
            scope=effective_scope,
            runtime_vars=vars,
        )
        request = self._run_lifecycle_hooks("before_request", request)
        request = _require_lifecycle_payload(
            "before_request", request, ModelRequestContext
        )
        model_execution_params = request.to_parameters()
        if self.config.get("verbose", False):
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        emit_event(
            EventType.MODEL_REQUEST,
            {"message_count": len(model_execution_params.get("messages") or [])},
        )
        response = self.generator(**model_execution_params)
        if not isinstance(response, ModelStreamResponse):
            response_context = self._run_lifecycle_hooks(
                "after_response",
                ModelResponseContext(
                    scope=effective_scope,
                    vars=vars,
                    response=response,
                    request=request,
                ),
            )
            response_context = _require_lifecycle_payload(
                "after_response", response_context, ModelResponseContext
            )
            response = response_context.response
        emit_model_response_events(response, scope=scope)
        return response

    async def _aexecute_model(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        scope: Optional[ExecutionScope] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        self._raise_if_background_task_interrupted()
        decision = await self._aresolve_continuation("before_request", messages, vars)
        if decision.action == "return":
            return _TerminalResponse(decision)
        effective_scope = scope or get_execution_context()["scope"]
        conversation = await self._arun_lifecycle_hooks(
            "transform_context",
            ConversationContext(
                scope=effective_scope,
                vars=vars,
                messages=messages,
            ),
        )
        conversation = _require_lifecycle_payload(
            "transform_context", conversation, ConversationContext
        )
        model_messages = await self._abuild_model_messages(
            conversation.messages,
            vars=vars,
            scope=effective_scope,
        )
        model_execution_params = self._prepare_model_execution(
            messages=conversation.messages,
            prefilling=prefilling,
            model_preference=model_preference,
            vars=vars,
            tool_filter=tool_filter,
            model_messages=model_messages,
            scope=effective_scope,
            transform_tool_catalog=False,
            transform_system_prompt=False,
        )
        model_execution_params = await self._atransform_model_tool_catalog(
            model_execution_params,
            vars=vars,
            scope=effective_scope,
        )
        model_execution_params.system_prompt = self._build_system_prompt(
            vars=vars,
            tool_catalog=model_execution_params.tool_catalog,
            apply_hooks=False,
        )
        model_execution_params = await self._atransform_model_system_prompt(
            model_execution_params,
            vars=vars,
            scope=effective_scope,
        )
        compacted = await self._amaybe_compact_model_context(
            conversation.messages,
            model_execution_params.messages,
            system_prompt=model_execution_params.system_prompt,
            tool_catalog=model_execution_params.tool_catalog,
            model_preference=model_preference,
            vars=vars,
            scope=effective_scope,
        )
        if compacted:
            model_execution_params.messages = await self._abuild_model_messages(
                conversation.messages,
                vars=vars,
                scope=effective_scope,
                drain_notifications=False,
            )
        request = ModelRequestContext.from_parameters(
            model_execution_params,
            scope=effective_scope,
            runtime_vars=vars,
        )
        request = await self._arun_lifecycle_hooks("before_request", request)
        request = _require_lifecycle_payload(
            "before_request", request, ModelRequestContext
        )
        model_execution_params = request.to_parameters()
        if self.config.get("verbose", False):
            cprint(f"[{self.name}][call_model]", bc="br1", ls="b")
        emit_event(
            EventType.MODEL_REQUEST,
            {"message_count": len(model_execution_params.get("messages") or [])},
        )
        response = await await_with_abort(
            self.generator.acall(**model_execution_params),
            (scope or get_execution_context()["scope"]).abort_signal,
        )
        if not isinstance(response, ModelStreamResponse):
            response_context = await self._arun_lifecycle_hooks(
                "after_response",
                ModelResponseContext(
                    scope=effective_scope,
                    vars=vars,
                    response=response,
                    request=request,
                ),
            )
            response_context = _require_lifecycle_payload(
                "after_response", response_context, ModelResponseContext
            )
            response = response_context.response
        emit_model_response_events(response, scope=scope)
        return response

    def warmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
        background: bool = False,
    ):
        """Warm the provider cache for the rendered system prompt and tools.

        This bypasses task messages, chat history, checkpoint stores and response
        parsing. Warmup should only include stable prompt prefixes; dynamic user
        content would reduce cache hits for the real request.
        """
        if background:
            detached(
                self._warmup_system_prompt,
                vars=vars,
                tool_filter=tool_filter,
                model_preference=model_preference,
            )
            return None
        return self._warmup_system_prompt(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )

    async def awarmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
        background: bool = False,
    ):
        """Async counterpart for warming the provider system prompt cache."""
        if background:
            await adetached(
                self._awarmup_system_prompt,
                vars=vars,
                tool_filter=tool_filter,
                model_preference=model_preference,
            )
            return None
        return await self._awarmup_system_prompt(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )

    def _warmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ):
        params = self._prepare_warmup_execution(
            vars=vars,
            tool_filter=tool_filter,
            model_preference=model_preference,
        )
        return self.model.warmup_system_prompt(**params)

    async def _awarmup_system_prompt(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ):
        model_execution_params = self._prepare_model_execution(
            messages=[],
            vars=vars or {},
            model_preference=model_preference,
            tool_filter=tool_filter,
            transform_tool_catalog=False,
            transform_system_prompt=False,
        )
        effective_scope = get_execution_context()["scope"]
        model_execution_params = await self._atransform_model_tool_catalog(
            model_execution_params,
            vars=vars or {},
            scope=effective_scope,
        )
        model_execution_params.system_prompt = self._build_system_prompt(
            vars=vars or {},
            tool_catalog=model_execution_params.tool_catalog,
            apply_hooks=False,
        )
        model_execution_params = await self._atransform_model_system_prompt(
            model_execution_params,
            vars=vars or {},
            scope=effective_scope,
        )
        params = dotdict(
            system_prompt=model_execution_params.system_prompt,
            tool_catalog=model_execution_params.tool_catalog,
            **(
                {"model_preference": model_execution_params.model_preference}
                if model_preference
                else {}
            ),
        )
        return await self.model.awarmup_system_prompt(**params)

    async def _atransform_model_system_prompt(
        self,
        model_execution_params: dotdict,
        *,
        vars: Mapping[str, Any],
        scope: Optional[ExecutionScope] = None,
    ) -> dotdict:
        prompt_ctx = await self._arun_lifecycle_hooks(
            "transform_system_prompt",
            ModelContext(
                system_prompt=model_execution_params.system_prompt or "",
                scope=scope or get_execution_context()["scope"],
                vars=vars,
                tool_catalog=model_execution_params.tool_catalog,
            ),
        )
        prompt_ctx = _require_lifecycle_payload(
            "transform_system_prompt", prompt_ctx, ModelContext
        )
        model_execution_params.system_prompt = prompt_ctx.system_prompt or None
        return model_execution_params

    def _prepare_warmup_execution(
        self,
        *,
        vars: Optional[Mapping[str, Any]] = None,
        tool_filter: Optional[ToolFilter] = None,
        model_preference: Optional[str] = None,
    ) -> Mapping[str, Any]:
        model_execution_params = self._prepare_model_execution(
            messages=[],
            vars=vars or {},
            model_preference=model_preference,
            tool_filter=tool_filter,
        )
        return dotdict(
            system_prompt=model_execution_params.system_prompt,
            tool_catalog=model_execution_params.tool_catalog,
            **(
                {"model_preference": model_execution_params.model_preference}
                if model_preference
                else {}
            ),
        )

    def _transform_model_tool_catalog(
        self,
        tool_catalog: ToolCatalogView,
        *,
        messages: Any,
        vars: Mapping[str, Any],
        scope: ExecutionScope,
    ) -> ToolCatalogView:
        catalog_context = self._run_lifecycle_hooks(
            "transform_tool_catalog",
            ToolCatalogContext(
                scope=scope,
                vars=vars,
                catalog=tool_catalog,
                messages=messages,
            ),
        )
        catalog_context = _require_lifecycle_payload(
            "transform_tool_catalog", catalog_context, ToolCatalogContext
        )
        if not isinstance(catalog_context.catalog, ToolCatalogView):
            raise TypeError("ToolCatalogContext.catalog must be a ToolCatalogView")
        return catalog_context.catalog

    async def _atransform_model_tool_catalog(
        self,
        model_execution_params: dotdict,
        *,
        vars: Mapping[str, Any],
        scope: ExecutionScope,
    ) -> dotdict:
        tool_catalog = model_execution_params.tool_catalog
        if tool_catalog is None:
            return model_execution_params
        catalog_context = await self._arun_lifecycle_hooks(
            "transform_tool_catalog",
            ToolCatalogContext(
                scope=scope,
                vars=vars,
                catalog=tool_catalog,
                messages=model_execution_params.messages,
            ),
        )
        catalog_context = _require_lifecycle_payload(
            "transform_tool_catalog", catalog_context, ToolCatalogContext
        )
        if not isinstance(catalog_context.catalog, ToolCatalogView):
            raise TypeError("ToolCatalogContext.catalog must be a ToolCatalogView")
        transformed = catalog_context.catalog
        model_execution_params.tool_catalog = (
            transformed if transformed.visible_entries() else None
        )
        return model_execution_params

    def _build_system_prompt(
        self,
        *,
        vars: Mapping[str, Any],
        tool_catalog: ToolCatalogView | None,
        apply_hooks: bool,
    ) -> str | None:
        system_prompt = self.get_system_prompt(
            vars,
            tool_catalog=tool_catalog,
            _apply_hooks=apply_hooks,
        )
        portable_schemas = tool_catalog.portable_schemas() if tool_catalog else []
        if is_subclass_of(self.generation_schema, ToolFlowControl) and portable_schemas:
            tools_template = self.generation_schema.tools_template
            inputs = {
                "tool_schemas": portable_schemas,
                "tool_choice": (
                    tool_catalog.choice.name
                    if tool_catalog.choice.mode == "tool"
                    else tool_catalog.choice.mode
                ),
            }
            flow_control_tools = self._format_template(inputs, tools_template)
            system_prompt = (
                f"{flow_control_tools}\n\n{system_prompt}"
                if system_prompt
                else flow_control_tools
            )
        return system_prompt or None

    def _prepare_model_execution(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        *,
        prefilling: Optional[str] = None,
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        drain_notifications: bool = True,
        scope: Optional[ExecutionScope] = None,
        model_messages: Any = None,
        transform_tool_catalog: bool = True,
        transform_system_prompt: bool = True,
    ) -> Mapping[str, Any]:
        effective_scope = scope or get_execution_context()["scope"]
        if model_messages is None:
            model_messages = self._build_model_messages(
                messages,
                vars=vars,
                scope=effective_scope,
                drain_notifications=drain_notifications,
            )

        tool_catalog = self.tool_library.get_tool_catalog_view(
            model_messages if isinstance(model_messages, ChatMessages) else None,
            thread_id=(
                effective_scope.thread_id or f"{self.tool_library.name}:unscoped"
            ),
        )
        tool_entries = list(tool_catalog.tool_entries())

        tool_choice = self.config.get("tool_choice")

        if tool_filter is not None and tool_entries:
            tool_entries = self._apply_tool_filter(tool_entries, tool_filter)

        tool_catalog = tool_catalog.with_tools(entry.name for entry in tool_entries)
        tool_catalog = tool_catalog.with_choice(
            self._resolve_tool_choice(tool_choice, tool_entries)
        )
        if transform_tool_catalog:
            tool_catalog = self._transform_model_tool_catalog(
                tool_catalog,
                messages=model_messages,
                vars=vars,
                scope=effective_scope,
            )
        tool_catalog = tool_catalog if tool_catalog.visible_entries() else None
        system_prompt = self._build_system_prompt(
            vars=vars,
            tool_catalog=tool_catalog,
            apply_hooks=transform_system_prompt,
        )

        model_execution_params = dotdict(
            messages=model_messages,
            system_prompt=system_prompt,
            prefilling=prefilling,
            stream=self.config.get("stream", False),
            tool_catalog=tool_catalog,
            generation_schema=self.generation_schema,
        )

        if model_preference:
            model_execution_params.model_preference = model_preference

        return model_execution_params

    # --- Response Processing ---

    def _ensure_stream_response_ready(
        self, model_response: ModelStreamResponse
    ) -> None:
        if model_response.response_type is not None:
            return

        error = getattr(model_response, "error", None)
        if error is not None:
            raise error

        raise RuntimeError("Model stream ended before producing a response type.")

    def _process_model_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        scope: Optional[ExecutionScope] = None,
    ) -> Union[str, Mapping[str, Any], Message, ModelStreamResponse]:
        if isinstance(model_response, ModelStreamResponse):
            wait_for_event(model_response._response_type_event)
            self._ensure_stream_response_ready(model_response)
            if model_response.response_type != "tool_call":
                self._checkpoint_save(messages, vars, status="streaming")
                self._attach_stream_checkpoint_finalizer(
                    model_response,
                    messages,
                    vars,
                )
                return self._prepare_response(
                    model_response,
                    model_response.response_type,
                    messages,
                    message,
                    vars,
                    model_response.reasoning,
                )

        if "tool_call" in model_response.response_type:
            model_response, messages = self._process_tool_call_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )
        elif not isinstance(model_response, _TerminalResponse) and is_subclass_of(
            self.generation_schema, ToolFlowControl
        ):
            model_response, messages = self._process_tool_flow_control_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
            reasoning = model_response.reasoning
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"
            reasoning = None

        response_item_start = len(messages) if isinstance(messages, ChatMessages) else 0
        self._append_response_to_chat_messages(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
            reasoning=reasoning,
            history_items=getattr(model_response, "history_items", None),
        )
        attach_response_metadata(
            messages,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
            after_index=response_item_start,
        )
        if response_type not in self._supported_outputs:
            raise ValueError(f"Unsupported `response_type={response_type}`")
        response = self._prepare_response(
            raw_response, response_type, messages, message, vars, reasoning
        )
        try:
            run_end = self._run_run_end_hook(
                "before_run_end",
                self._run_end_context(
                    outcome="completed",
                    messages=messages,
                    vars=vars,
                    scope=scope,
                    output=response,
                ),
            )
        except Exception as error:
            raise _BeforeRunEndHookError(error) from error
        self._finalize_chat_turn(run_end.messages, raw_response)
        self._checkpoint_save(run_end.messages, vars, status="completed")
        run_end = self._run_after_run_end_hook(run_end)
        return run_end.output

    async def _aprocess_model_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
        scope: Optional[ExecutionScope] = None,
    ) -> Union[str, Mapping[str, Any], Message, ModelStreamResponse]:
        if isinstance(model_response, ModelStreamResponse):
            await await_for_event(model_response._response_type_event)
            self._ensure_stream_response_ready(model_response)
            if model_response.response_type != "tool_call":
                await self._acheckpoint_save(messages, vars, status="streaming")
                self._attach_stream_checkpoint_finalizer(
                    model_response,
                    messages,
                    vars,
                    async_mode=True,
                )
                return self._prepare_response(
                    model_response,
                    model_response.response_type,
                    messages,
                    message,
                    vars,
                    model_response.reasoning,
                )

        if "tool_call" in model_response.response_type:
            model_response, messages = await self._aprocess_tool_call_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )
        elif not isinstance(model_response, _TerminalResponse) and is_subclass_of(
            self.generation_schema, ToolFlowControl
        ):
            (
                model_response,
                messages,
            ) = await self._aprocess_tool_flow_control_response(
                message,
                model_response,
                messages,
                vars,
                model_preference,
                tool_filter,
            )

        if isinstance(model_response, (ModelResponse, ModelStreamResponse)):
            raw_response = self._extract_raw_response(model_response)
            response_type = model_response.response_type
            reasoning = model_response.reasoning
        else:  # returns tool result as response or tool call as response
            raw_response = model_response
            response_type = "tool_responses"
            reasoning = None

        response_item_start = len(messages) if isinstance(messages, ChatMessages) else 0
        self._append_response_to_chat_messages(
            messages,
            raw_response,
            response_type,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
            reasoning=reasoning,
            history_items=getattr(model_response, "history_items", None),
        )
        attach_response_metadata(
            messages,
            getattr(model_response, "metadata", None)
            if isinstance(model_response, (ModelResponse, ModelStreamResponse))
            else None,
            after_index=response_item_start,
        )
        if response_type not in self._supported_outputs:
            raise ValueError(f"Unsupported `response_type={response_type}`")
        response = self._prepare_response(
            raw_response, response_type, messages, message, vars, reasoning
        )
        try:
            run_end = await self._arun_run_end_hook(
                "before_run_end",
                self._run_end_context(
                    outcome="completed",
                    messages=messages,
                    vars=vars,
                    scope=scope,
                    output=response,
                ),
            )
        except Exception as error:
            raise _BeforeRunEndHookError(error) from error
        self._finalize_chat_turn(run_end.messages, raw_response)
        await self._acheckpoint_save(run_end.messages, vars, status="completed")
        run_end = await self._arun_after_run_end_hook(run_end)
        return run_end.output

    # --- Tool Processing ---

    def _process_tool_flow_control_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Handle tool flow control responses using the ToolFlowControl interface."""
        flow_control = self.generation_schema
        while True:
            if isinstance(model_response, _TerminalResponse):
                return model_response, messages
            response_item_start = (
                len(messages) if isinstance(messages, ChatMessages) else 0
            )
            raw_response = self._extract_raw_response(model_response)

            # Use ToolFlowControl interface via generation_schema
            flow_result = flow_control.extract_flow_result(raw_response)

            if flow_result.is_complete:
                if flow_result.final_response is not None:
                    model_response.data = flow_result.final_response
                return model_response, messages

            if self.config.get("verbose", False) and flow_result.reasoning:
                cprint(
                    f"[{self.name}][tool_calls_reasoning] {flow_result.reasoning}",
                    bc="br2",
                    ls="b",
                )

            if flow_result.tool_calls:
                intents = self._flow_tool_intents(flow_result.tool_calls)
                self._validate_context_scope_intents(intents)
                outcomes = self._process_tool_intents(intents, message, messages, vars)
                tool_results = ToolResponses.from_outcomes(intents, outcomes)

                # Use interface to inject results
                raw_response = flow_control.inject_results(raw_response, tool_results)

                # Use interface to build history
                messages = flow_control.build_history(raw_response, messages)
                attach_response_metadata(
                    messages,
                    getattr(model_response, "metadata", None),
                    after_index=response_item_start,
                )
                self._apply_context_scope_outcomes(messages, outcomes)
                decision = self._resolve_continuation(
                    "after_tools", messages, vars, intents, outcomes
                )
                if decision.action == "return":
                    return _TerminalResponse(decision), messages
                feedback = self._resolve_tool_feedback(
                    intents,
                    outcomes,
                    messages=messages,
                    vars=vars,
                    reasoning=flow_result.reasoning,
                )
                if feedback.action == "return":
                    return feedback.output, messages
                self._drain_inbox_into_messages(messages, vars=vars)
                self._checkpoint_save(messages, vars)

            model_response = self._execute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    async def _aprocess_tool_flow_control_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Async version of _process_tool_flow_control_response.
        Handle tool flow control responses using the ToolFlowControl interface.
        """
        flow_control = self.generation_schema
        while True:
            if isinstance(model_response, _TerminalResponse):
                return model_response, messages
            response_item_start = (
                len(messages) if isinstance(messages, ChatMessages) else 0
            )
            raw_response = self._extract_raw_response(model_response)

            # Use ToolFlowControl interface via generation_schema (async)
            flow_result = await flow_control.aextract_flow_result(raw_response)

            if flow_result.is_complete:
                if flow_result.final_response is not None:
                    model_response.data = flow_result.final_response
                return model_response, messages

            if self.config.get("verbose", False) and flow_result.reasoning:
                cprint(
                    f"[{self.name}][tool_calls_reasoning] {flow_result.reasoning}",
                    bc="br2",
                    ls="b",
                )

            if flow_result.tool_calls:
                intents = self._flow_tool_intents(flow_result.tool_calls)
                self._validate_context_scope_intents(intents)
                outcomes = await self._aprocess_tool_intents(
                    intents, message, messages, vars
                )
                tool_results = ToolResponses.from_outcomes(intents, outcomes)

                # Use interface to inject results (async version)
                raw_response = await flow_control.ainject_results(
                    raw_response, tool_results
                )

                # Use interface to build history (async version)
                messages = await flow_control.abuild_history(raw_response, messages)
                attach_response_metadata(
                    messages,
                    getattr(model_response, "metadata", None),
                    after_index=response_item_start,
                )
                self._apply_context_scope_outcomes(messages, outcomes)
                decision = await self._aresolve_continuation(
                    "after_tools", messages, vars, intents, outcomes
                )
                if decision.action == "return":
                    return _TerminalResponse(decision), messages
                feedback = await self._aresolve_tool_feedback(
                    intents,
                    outcomes,
                    messages=messages,
                    vars=vars,
                    reasoning=flow_result.reasoning,
                )
                if feedback.action == "return":
                    return feedback.output, messages
                await self._adrain_inbox_into_messages(messages, vars=vars)
                await self._acheckpoint_save(messages, vars)

            model_response = await self._aexecute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    def _process_tool_call_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """ToolCall example:
        [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        while True:
            if model_response.response_type == "tool_call":
                response_item_start = (
                    len(messages) if isinstance(messages, ChatMessages) else 0
                )
                raw_response = model_response.data
                reasoning = model_response.reasoning
                appended_item_types = self._append_tool_model_history(
                    messages, model_response
                )
                attach_response_metadata(
                    messages,
                    getattr(model_response, "metadata", None),
                    after_index=response_item_start,
                )
                if "reasoning" in appended_item_types:
                    raw_response.reasoning = None

                if self.config.get("verbose", False):
                    if reasoning:
                        repr_str = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr_str, bc="br2", ls="b")

                tool_intents = model_response.get_tool_intents()
                self._validate_context_scope_intents(tool_intents)
                try:
                    tool_outcomes = self._process_tool_intents(
                        tool_intents, message, messages, vars
                    )
                except TaskInterruptRequestedError as exc:
                    self._append_interrupted_tool_response_messages(
                        messages,
                        model_response,
                        reason=str(exc),
                    )
                    raise

                tool_responses_message = model_response.render_tool_outcomes(
                    tool_outcomes
                )
                self._extend_tool_response_history(messages, tool_responses_message)
                self._apply_context_scope_outcomes(messages, tool_outcomes)
                decision = self._resolve_continuation(
                    "after_tools", messages, vars, tool_intents, tool_outcomes
                )
                if decision.action == "return":
                    return _TerminalResponse(decision), messages
                feedback = self._resolve_tool_feedback(
                    tool_intents,
                    tool_outcomes,
                    messages=messages,
                    vars=vars,
                    reasoning=reasoning,
                )
                if feedback.action == "return":
                    return feedback.output, messages

                self._drain_inbox_into_messages(messages, vars=vars)
                self._checkpoint_save(messages, vars)
            else:
                return model_response, messages

            model_response = self._execute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    async def _aprocess_tool_call_response(
        self,
        message: Union[str, Mapping[str, Any], Message],
        model_response: Union[ModelResponse, ModelStreamResponse],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        model_preference: Optional[str] = None,
        tool_filter: Optional[ToolFilter] = None,
    ) -> Tuple[
        Union[str, Mapping[str, Any], ModelStreamResponse],
        Union[ChatMessages, List[Mapping[str, Any]]],
    ]:
        """Async version of _process_tool_call_response.
        ToolCall example: [{'role': 'assistant', 'tool_responses': [{'id': 'call_1YL',
        'type': 'function', 'function': {'arguments': '{"order_id":"order_12345"}',
        'name': 'get_delivery_date'}}]}, {'role': 'tool', 'tool_call_id': 'call_HA',
        'content': '2024-10-15'}].
        """
        while True:
            if model_response.response_type == "tool_call":
                if isinstance(model_response, ModelStreamResponse):
                    await self._aconsume_event_response(
                        model_response,
                        emit_content=False,
                    )
                response_item_start = (
                    len(messages) if isinstance(messages, ChatMessages) else 0
                )
                raw_response = model_response.data
                reasoning = model_response.reasoning
                appended_item_types = self._append_tool_model_history(
                    messages, model_response
                )
                attach_response_metadata(
                    messages,
                    getattr(model_response, "metadata", None),
                    after_index=response_item_start,
                )
                if "reasoning" in appended_item_types:
                    raw_response.reasoning = None

                if self.config.get("verbose", False):
                    if reasoning:
                        repr_str = f"[{self.name}][tool_calls_reasoning] {reasoning}"
                        cprint(repr_str, bc="br2", ls="b")

                tool_intents = model_response.get_tool_intents()
                self._validate_context_scope_intents(tool_intents)
                try:
                    tool_outcomes = await self._aprocess_tool_intents(
                        tool_intents, message, messages, vars
                    )
                except TaskInterruptRequestedError as exc:
                    self._append_interrupted_tool_response_messages(
                        messages,
                        model_response,
                        reason=str(exc),
                    )
                    raise

                tool_responses_message = model_response.render_tool_outcomes(
                    tool_outcomes
                )
                self._extend_tool_response_history(messages, tool_responses_message)
                self._apply_context_scope_outcomes(messages, tool_outcomes)
                decision = await self._aresolve_continuation(
                    "after_tools", messages, vars, tool_intents, tool_outcomes
                )
                if decision.action == "return":
                    return _TerminalResponse(decision), messages
                feedback = await self._aresolve_tool_feedback(
                    tool_intents,
                    tool_outcomes,
                    messages=messages,
                    vars=vars,
                    reasoning=reasoning,
                )
                if feedback.action == "return":
                    return feedback.output, messages

                await self._adrain_inbox_into_messages(messages, vars=vars)
                await self._acheckpoint_save(messages, vars)
            else:
                return model_response, messages

            model_response = await self._aexecute_model(
                messages=messages,
                model_preference=model_preference,
                vars=vars,
                tool_filter=tool_filter,
            )

    @staticmethod
    def _flow_tool_intents(calls) -> tuple[ToolIntent, ...]:
        return tuple(
            ToolIntent(id=call_id, name=name, arguments=coerce_tool_params(name, args))
            for call_id, name, args in calls
        )

    def _process_tool_intents(
        self,
        intents: tuple[ToolIntent, ...],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> tuple[ToolOutcome, ...]:
        self.tool_library.set_lifecycle_owner(self)
        self._log_tool_intents(intents)
        outcomes = self.tool_library.execute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        self._log_tool_outcomes(outcomes)
        return outcomes

    async def _aprocess_tool_intents(
        self,
        intents: tuple[ToolIntent, ...],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> tuple[ToolOutcome, ...]:
        self.tool_library.set_lifecycle_owner(self)
        self._log_tool_intents(intents)
        outcomes = await self.tool_library.aexecute_intents(
            intents,
            message=message,
            messages=messages,
            vars=vars,
        )
        self._log_tool_outcomes(outcomes)
        return outcomes

    def _log_tool_intents(self, intents: tuple[ToolIntent, ...]) -> None:
        if not self.config.get("verbose", False):
            return
        for intent in intents:
            repr_str = f"[{self.name}][tool_call] {intent.name}: {intent.arguments}"
            cprint(repr_str, bc="br2", ls="b")

    def _log_tool_outcomes(self, outcomes: tuple[ToolOutcome, ...]) -> None:
        if not self.config.get("verbose", False):
            return
        repr_str = f"[{self.name}][tool_responses]"
        cprint(repr_str, bc="br1", ls="b")
        for outcome in outcomes:
            result = (
                outcome.error.message if outcome.error is not None else outcome.result
            )
            repr_str = (
                f"[{self.name}][tool_response] {outcome.tool_name}: {result or ''}"
            )
            cprint(repr_str, ls="b")

    @staticmethod
    def _validate_tool_feedback_context(
        context: Any,
    ) -> ToolFeedbackContext:
        if not isinstance(context, ToolFeedbackContext):
            raise TypeError(
                "Agent `resolve_tool_feedback` hooks must return ToolFeedbackContext"
            )
        if context.action not in {"continue", "return"}:
            raise ValueError(
                "ToolFeedbackContext.action must be `continue` or `return`"
            )
        return context

    def _tool_feedback_context(
        self,
        intents: tuple[ToolIntent, ...],
        outcomes: tuple[ToolOutcome, ...],
        *,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        reasoning: str | None,
    ) -> ToolFeedbackContext:
        return ToolFeedbackContext(
            scope=get_execution_context()["scope"],
            vars=vars,
            intents=intents,
            outcomes=outcomes,
            messages=messages,
            reasoning=reasoning,
        )

    def _resolve_tool_feedback(
        self,
        intents: tuple[ToolIntent, ...],
        outcomes: tuple[ToolOutcome, ...],
        *,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        reasoning: str | None,
    ) -> ToolFeedbackContext:
        context = self._run_lifecycle_hooks(
            "resolve_tool_feedback",
            self._tool_feedback_context(
                intents,
                outcomes,
                messages=messages,
                vars=vars,
                reasoning=reasoning,
            ),
            stop_when=lambda current: getattr(current, "action", None) != "continue",
        )
        return self._validate_tool_feedback_context(context)

    async def _aresolve_tool_feedback(
        self,
        intents: tuple[ToolIntent, ...],
        outcomes: tuple[ToolOutcome, ...],
        *,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
        reasoning: str | None,
    ) -> ToolFeedbackContext:
        context = await self._arun_lifecycle_hooks(
            "resolve_tool_feedback",
            self._tool_feedback_context(
                intents,
                outcomes,
                messages=messages,
                vars=vars,
                reasoning=reasoning,
            ),
            stop_when=lambda current: getattr(current, "action", None) != "continue",
        )
        return self._validate_tool_feedback_context(context)

    def _process_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        self.tool_library.set_lifecycle_owner(self)
        if self.config.get("verbose", False):
            for call in tool_callings:
                repr_str = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr_str, bc="br2", ls="b")
        tool_results = self.tool_library(
            tool_callings=tool_callings,
            message=message,
            messages=messages,
            vars=vars,
        )
        if self.config.get("verbose", False):
            repr_str = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr_str += " return directly"
            cprint(repr_str, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr_str = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr_str, ls="b")
        return tool_results

    async def _aprocess_tool_call(
        self,
        tool_callings: Mapping[str, Any],
        message: Union[str, Mapping[str, Any], Message],
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        vars: Mapping[str, Any],
    ) -> ToolResponses:
        """Async version of _process_tool_call."""
        self.tool_library.set_lifecycle_owner(self)
        if self.config.get("verbose", False):
            for call in tool_callings:
                repr_str = f"[{self.name}][tool_call] {call[1]}: {call[2]}"
                cprint(repr_str, bc="br2", ls="b")
        tool_results = await self.tool_library.acall(
            tool_callings=tool_callings,
            message=message,
            messages=messages,
            vars=vars,
        )
        if self.config.get("verbose", False):
            repr_str = f"[{self.name}][tool_responses]"
            if tool_results.return_directly:
                repr_str += " return directly"
            cprint(repr_str, bc="br1", ls="b")
            for call in tool_results.tool_calls:
                result = call.result or call.error or ""
                repr_str = f"[{self.name}][tool_response] {call.name}: {result}"
                cprint(repr_str, ls="b")
        return tool_results

    def _apply_reasoning_in_response(self, raw_response, reasoning):
        if self.config.get("reasoning_in_response", False) and reasoning is not None:
            return dotdict(answer=raw_response, reasoning=reasoning)
        return raw_response

    def _prepare_response(
        self,
        raw_response: Union[str, Mapping[str, Any], ModelStreamResponse],
        response_type: str,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        message: Union[str, Mapping[str, Any], Message],
        vars: Mapping[str, Any],
        reasoning: Optional[str] = None,
    ) -> Union[str, Mapping[str, Any], ModelStreamResponse]:
        formatted_response = None
        if not isinstance(raw_response, ModelStreamResponse):
            raw_response = self._apply_reasoning_in_response(raw_response, reasoning)

            if response_type == "text_generation" or "structured" in response_type:
                if self.config.get("verbose", False):
                    if reasoning:
                        cprint(
                            f"[{self.name}][reasoning] {reasoning}", bc="br2", ls="b"
                        )
                    cprint(f"[{self.name}][response] {raw_response}", bc="y", ls="b")
                if self.templates.get("response"):
                    if isinstance(raw_response, str):
                        pre_response = self._format_response_template(vars)
                        formatted_response = self._format_template(
                            raw_response, pre_response
                        )
                    elif isinstance(raw_response, dict):
                        raw_response.update(vars)
                        formatted_response = self._format_response_template(
                            raw_response
                        )

        result = formatted_response or raw_response
        if self.config.get("return_messages", False):
            if response_type == "tool_responses":
                result.messages = messages
            else:
                result = dotdict(response=result, messages=messages)
        return self._define_response_mode(result, message)

    # --- Task Preparation ---
