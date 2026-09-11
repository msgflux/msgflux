import asyncio
from copy import deepcopy
from types import SimpleNamespace

import msgspec
import pytest

from msgflux.nn.modules.tool import LocalTool, MCPTool
from msgflux.nn.modules.tool_runtime import (
    BackgroundDispatch,
    ContextBinding,
    ContextSpec,
    DetachedDispatch,
    DispatchRequest,
    DispatchSpec,
    FeedbackSpec,
    ForegroundDispatch,
    LoadingSpec,
    NativeToolBinding,
    RuntimeContextProvider,
    ToolContextProvider,
    ToolDefinition,
    ToolDispatch,
    ToolError,
    ToolExecutionPlan,
    ToolExtension,
    ToolExtensionRegistry,
    ToolIntent,
    ToolOutcome,
    ToolRef,
    ToolRuntimeContext,
)


def sample_tool(value: int) -> int:
    return value * 2


def make_executor(impl=sample_tool):
    return LocalTool(
        name="sample_tool",
        description="Double a value.",
        annotations={"value": int, "return": int},
        tool_config={},
        impl=impl,
    )


def make_definition(**overrides):
    values = {
        "name": "sample_tool",
        "executor": make_executor(),
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        "description": "Double a value.",
        "annotations": {"value": int, "return": int},
    }
    values.update(overrides)
    return ToolDefinition(**values)


def make_intent(**overrides):
    values = {
        "id": "call_1",
        "name": "sample_tool",
        "arguments": {"value": 2},
    }
    values.update(overrides)
    return ToolIntent(**values)


def make_plan(**definition_overrides):
    definition = make_definition(**definition_overrides)
    intent = make_intent(name=definition.name)
    return ToolExecutionPlan(
        intent=intent,
        definition=definition,
        visible_arguments=intent.arguments,
    )


def test_tool_definition_normalizes_open_specs_and_context_bindings():
    definition = make_definition(
        dispatch="queue",
        feedback="direct",
        context=(
            "messages",
            ContextBinding(source="vars", options={"select": ["tenant"]}),
        ),
    )

    assert definition.dispatch == DispatchSpec(name="queue")
    assert definition.feedback == FeedbackSpec(name="direct")
    assert [binding.parameter for binding in definition.context.bindings] == [
        "messages",
        "vars",
    ]
    assert definition.context.bindings[1].options == {"select": ["tenant"]}


def test_tool_definition_groups_loading_context_and_native_bindings():
    binding = NativeToolBinding(
        provider="openai",
        api_mode="responses",
        kind="shell",
        execution="client",
        options={"environment": {"type": "local"}},
    )
    definition = make_definition(
        context=ContextSpec(bindings=("handle",)),
        loading=LoadingSpec(deferred=True),
        native_bindings=(binding,),
    )

    assert definition.context == ContextSpec(
        bindings=(ContextBinding(source="handle"),)
    )
    assert definition.loading == LoadingSpec(deferred=True)
    assert definition.native_bindings == (binding,)


def test_tool_definition_rejects_duplicate_context_parameters():
    with pytest.raises(ValueError, match="unique parameters"):
        make_definition(
            context=(
                ContextBinding(source="messages", parameter="context"),
                ContextBinding(source="vars", parameter="context"),
            )
        )


def test_tool_definition_accepts_local_and_mcp_execution_adapters():
    class MCPClient:
        async def call_tool(self, _name, _arguments):
            raise AssertionError("The contract test must not execute the tool")

    local_definition = make_definition()
    mcp_info = SimpleNamespace(
        name="lookup",
        description="Look up a remote value.",
        inputSchema={"type": "object", "properties": {}},
    )
    mcp_executor = MCPTool(
        name="lookup",
        mcp_client=MCPClient(),
        mcp_tool_info=mcp_info,
        namespace="remote",
    )
    mcp_definition = ToolDefinition(
        name="remote__lookup",
        executor=mcp_executor,
        input_schema=mcp_info.inputSchema,
        description=mcp_info.description,
    )

    assert isinstance(local_definition.executor, LocalTool)
    assert mcp_definition.executor is mcp_executor


def test_tool_definition_rejects_raw_callable_before_adapter_compilation():
    with pytest.raises(TypeError, match="must provide synchronous"):
        make_definition(executor=sample_tool)


def test_tool_execution_plan_keeps_visible_and_runtime_arguments_separate():
    handle = object()
    plan = ToolExecutionPlan(
        intent=make_intent(),
        definition=make_definition(),
        visible_arguments={"value": 2},
        runtime_arguments={"handle": handle},
    )

    assert plan.visible_arguments == {"value": 2}
    assert set(plan.runtime_arguments) == {"handle"}
    assert plan.runtime_arguments["handle"] is handle
    assert set(plan.call_arguments) == {"value", "handle"}


def test_tool_execution_plan_rejects_visible_runtime_argument_collisions():
    with pytest.raises(ValueError, match="both visible and runtime"):
        ToolExecutionPlan(
            intent=make_intent(),
            definition=make_definition(),
            visible_arguments={"value": 2},
            runtime_arguments={"value": 3},
        )


def test_transport_contracts_round_trip_with_msgspec():
    ref = ToolRef(library_id="root_tools", tool_id="researcher")
    intent = ToolIntent(
        id="call_1",
        name=ref.tool_id,
        arguments={"message": "Investigate the outage."},
    )

    encoded_ref = msgspec.json.encode(ref)
    encoded_intent = msgspec.json.encode(intent)

    assert msgspec.json.decode(encoded_ref, type=ToolRef) == ref
    assert msgspec.json.decode(encoded_intent, type=ToolIntent) == intent


def test_tool_outcome_requires_structured_errors():
    intent = make_intent()

    with pytest.raises(ValueError, match="requires `error`"):
        ToolOutcome(
            intent_id=intent.id,
            tool_name=intent.name,
            status="blocked",
        )

    outcome = ToolOutcome(
        intent_id=intent.id,
        tool_name=intent.name,
        status="blocked",
        error=ToolError(code="policy_denied", message="Denied by policy."),
    )

    assert not outcome.ok
    assert outcome.error.code == "policy_denied"


def test_default_extension_registry_owns_open_dispatch_names():
    registry = ToolExtensionRegistry(install_defaults=True)

    assert isinstance(registry.get_dispatch("foreground"), ForegroundDispatch)
    assert isinstance(registry.get_dispatch("background"), BackgroundDispatch)
    assert isinstance(registry.get_dispatch("detached"), DetachedDispatch)
    assert isinstance(
        registry.get_context_provider("messages"),
        RuntimeContextProvider,
    )


def test_dispatch_registration_is_transactional_and_removable():
    class QueueDispatch(ToolDispatch):
        def __init__(self, extension_name="dispatch_queue"):
            super().__init__(extension_name, dispatch_name="queue")

        async def dispatch(self, request):
            return ToolOutcome.dispatched(request.plan.intent)

    registry = ToolExtensionRegistry()
    extension = QueueDispatch()
    handle = registry.register(extension)

    assert handle.active
    assert registry.get_dispatch("queue") is extension
    with pytest.raises(ValueError, match="already registered"):
        registry.register(QueueDispatch("another_queue"))

    handle.remove()

    assert not handle.active
    with pytest.raises(ValueError, match="not registered"):
        registry.get_dispatch("queue")


def test_extension_instance_has_exactly_one_registry_owner():
    extension = ForegroundDispatch()
    first = ToolExtensionRegistry([extension])
    second = ToolExtensionRegistry()

    with pytest.raises(ValueError, match="already registered on a registry"):
        second.register(extension)

    first.remove(extension.name)
    second.register(extension)

    assert extension.registry is second


def test_extension_registration_rolls_back_failed_setup():
    lifecycle = []

    class BrokenExtension(ToolExtension):
        def __init__(self):
            super().__init__("broken")

        def on_register(self, _registry):
            lifecycle.append("register")
            raise RuntimeError("setup failed")

        def on_remove(self, _registry):
            lifecycle.append("remove")

    extension = BrokenExtension()
    registry = ToolExtensionRegistry()

    with pytest.raises(RuntimeError, match="setup failed"):
        registry.register(extension)

    assert not registry.has("broken")
    assert lifecycle == ["register", "remove"]
    with pytest.raises(RuntimeError, match="not registered"):
        _ = extension.registry


@pytest.mark.asyncio
async def test_extension_registration_supports_async_setup_and_rollback():
    lifecycle = []

    class AsyncExtension(ToolExtension):
        def __init__(self):
            super().__init__("async_extension")

        async def aon_register(self, _registry):
            await asyncio.sleep(0)
            lifecycle.append("register")

        async def aon_remove(self, _registry):
            await asyncio.sleep(0)
            lifecycle.append("remove")

    registry = ToolExtensionRegistry()
    extension = AsyncExtension()
    handle = await registry.aregister(extension)

    assert handle.active
    assert lifecycle == ["register"]

    await handle.aremove()

    assert not handle.active
    assert lifecycle == ["register", "remove"]


@pytest.mark.asyncio
async def test_context_bindings_are_opt_in_selected_and_sequential():
    observed = []

    class TenantProvider(ToolContextProvider):
        def __init__(self):
            super().__init__("context_tenant", sources=("tenant",))

        async def resolve(self, request):
            observed.append(request.binding.source)
            await asyncio.sleep(0)
            return request.context.require("tenant")

    definition = make_definition(
        context=(
            ContextBinding(source="vars", options={"select": ("account",)}),
            ContextBinding(source="tenant", parameter="tenant_id"),
        )
    )
    intent = make_intent()
    runtime = ToolRuntimeContext(
        values={
            "messages": ["must not be injected"],
            "vars": {"account": "A-1", "secret": "hidden"},
            "tenant": "tenant-42",
        }
    )
    registry = ToolExtensionRegistry([RuntimeContextProvider(), TenantProvider()])

    resolved = await registry.resolve_context(definition, intent, runtime)

    assert resolved == {
        "vars": {"account": "A-1"},
        "tenant_id": "tenant-42",
    }
    assert "messages" not in resolved
    assert observed == ["tenant"]


@pytest.mark.asyncio
async def test_optional_context_binding_is_omitted_when_unavailable():
    definition = make_definition(
        context=(ContextBinding(source="messages", required=False),)
    )
    registry = ToolExtensionRegistry([RuntimeContextProvider()])

    resolved = await registry.resolve_context(
        definition,
        make_intent(),
        ToolRuntimeContext(),
    )

    assert resolved == {}


@pytest.mark.asyncio
async def test_foreground_dispatch_awaits_canonical_execution():
    plan = make_plan()
    calls = []

    async def execute(_plan=None):
        calls.append("execute")
        return ToolOutcome.completed(plan.intent, 4)

    registry = ToolExtensionRegistry([ForegroundDispatch()])
    outcome = await registry.dispatch(
        DispatchRequest(
            plan=plan,
            context=ToolRuntimeContext(),
            execute=execute,
        )
    )

    assert calls == ["execute"]
    assert outcome.result == 4


@pytest.mark.asyncio
async def test_detached_dispatch_returns_before_execution_settles():
    plan = make_plan(dispatch="detached")
    release = asyncio.Event()
    settled = asyncio.Event()

    async def execute(_plan=None):
        await release.wait()
        settled.set()
        return ToolOutcome.completed(plan.intent, 4)

    registry = ToolExtensionRegistry([DetachedDispatch()])
    outcome = await registry.dispatch(
        DispatchRequest(
            plan=plan,
            context=ToolRuntimeContext(),
            execute=execute,
        )
    )

    assert outcome.status == "dispatched"
    assert not settled.is_set()
    release.set()
    await asyncio.wait_for(settled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_background_dispatch_delegates_to_runtime_service():
    plan = make_plan(
        dispatch=DispatchSpec(name="background", options={"queue": "durable"})
    )
    requests = []

    class Scheduler:
        async def adispatch(self, request):
            requests.append(request)
            return ToolOutcome.dispatched(
                request.plan.intent,
                result={"task_id": "task_1"},
            )

    async def execute(_plan=None):
        raise AssertionError("background extension owns scheduling")

    registry = ToolExtensionRegistry([BackgroundDispatch()])
    outcome = await registry.dispatch(
        DispatchRequest(
            plan=plan,
            context=ToolRuntimeContext(values={"background_dispatcher": Scheduler()}),
            execute=execute,
        )
    )

    assert requests[0].plan.dispatch.options == {"queue": "durable"}
    assert outcome.result == {"task_id": "task_1"}


@pytest.mark.asyncio
async def test_custom_dispatch_is_a_first_class_extension():
    class QueueDispatch(ToolDispatch):
        def __init__(self):
            super().__init__("dispatch_queue", dispatch_name="queue")

        async def dispatch(self, request):
            return ToolOutcome.dispatched(
                request.plan.intent,
                result={"queue": request.plan.dispatch.options["name"]},
            )

    plan = make_plan(dispatch=DispatchSpec(name="queue", options={"name": "gpu"}))
    registry = ToolExtensionRegistry([QueueDispatch()])

    outcome = await registry.dispatch(
        DispatchRequest(
            plan=plan,
            context=ToolRuntimeContext(),
            execute=lambda _plan=None: None,
        )
    )

    assert outcome.result == {"queue": "gpu"}


def test_extension_registry_rebinds_extensions_after_deepcopy():
    registry = ToolExtensionRegistry(install_defaults=True)

    copied = deepcopy(registry)

    assert copied.get_dispatch("foreground").registry is copied
