import asyncio
import threading
from dataclasses import replace

import msgspec
import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import (
    Agent,
    AgentExtension,
    CompactionExtension,
    CompactionPolicy,
)
from msgflux.nn.hooks import (
    ConversationContext,
    Hook,
    ModelContext,
    ModelRequestContext,
    ModelResponseContext,
    NotificationContext,
    RunEndContext,
    ToolCatalogContext,
    ToolFeedbackContext,
)
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import AbortSignal, ExecutionScope
from msgflux.runtime.context import execution_context
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import FeedbackSpec, ToolIntent, ToolOutcome


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _RecordingModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _text_response("ok")

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return _text_response("ok")


class _CompactingModel(_RecordingModel):
    context_capacity = 100
    provider = "test"
    api_mode = "messages"
    model_id = "test-model"

    def __init__(self):
        super().__init__()
        self.compaction_calls = []

    def count_context_tokens(self, messages, *, system_prompt=None, tool_catalog=None):
        return ContextTokenEstimate(input_tokens=90, source="heuristic")

    async def acount_context_tokens(
        self, messages, *, system_prompt=None, tool_catalog=None
    ):
        return self.count_context_tokens(
            messages,
            system_prompt=system_prompt,
            tool_catalog=tool_catalog,
        )

    def compact_context(self, messages, *, system_prompt=None, native=True):
        self.compaction_calls.append((messages.to_items(), system_prompt))
        return ModelCompaction(
            format="messages",
            items=[{"role": "system", "content": "SUMMARY"}],
            provider=self.provider,
            api_mode=self.api_mode,
            model_id=self.model_id,
            usage={"input_tokens": 40, "output_tokens": 5},
        )

    async def acompact_context(self, messages, *, system_prompt=None, native=True):
        return self.compact_context(
            messages,
            system_prompt=system_prompt,
            native=native,
        )


class _PromptExtension(AgentExtension):
    def __init__(self):
        super().__init__("prompt")

    def hooks(self):
        def add_prompt(ctx: ModelContext):
            return replace(
                ctx,
                system_prompt=f"{ctx.system_prompt}\nextension prompt".strip(),
            )

        return (Hook(event="transform_system_prompt", handler=add_prompt),)

    def tools(self):
        def extension_tool(value: str) -> str:
            """Return the supplied value."""
            return value

        return (extension_tool,)


def _tool_feedback_values(*modes: str):
    intents = tuple(
        ToolIntent(id=f"call_{index}", name=f"tool_{index}")
        for index, _mode in enumerate(modes)
    )
    outcomes = tuple(
        ToolOutcome.completed(
            intent,
            f"result_{index}",
            feedback=FeedbackSpec(name=mode),
        )
        for index, (intent, mode) in enumerate(zip(intents, modes))
    )
    return intents, outcomes


def _history_with_completed_turn() -> ChatMessages:
    messages = ChatMessages(thread_id="thread_1", namespace="agent")
    messages.begin_turn(turn_id="old_turn")
    messages.add_user("Old question")
    messages.add_assistant("Old answer")
    messages.end_turn()
    return messages


def test_compaction_extension_appends_view_and_rebuilds_model_context():
    model = _CompactingModel()
    store = InMemoryCheckpointStore()
    messages = _history_with_completed_turn()
    agent = Agent(
        name="agent",
        model=model,
        system_prompt="Keep facts.",
        checkpoint_store=store,
        extensions=[
            CompactionExtension(
                CompactionPolicy(
                    trigger_ratio=0.8,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            )
        ],
    )

    output = agent(
        "New question",
        messages=messages,
        scope=ExecutionScope(
            thread_id="thread_1",
            namespace="agent",
            run_id="new_turn",
        ),
    )

    assert output == "ok"
    assert len(model.compaction_calls) == 1
    assert model.calls[0]["messages"].to_chatml() == [
        {"role": "system", "content": "SUMMARY"},
        {"role": "user", "content": "New question"},
    ]
    operation = messages.latest_compaction()
    assert (
        operation["compacted_through_item_id"]
        == messages.turns[0]["events"][-1]["item_id"]
    )
    assert operation["metadata"]["input_tokens_before"] == 90
    assert operation["metadata"]["usage"] == {
        "input_tokens": 40,
        "output_tokens": 5,
    }
    state = store.load_state("agent", "thread_1", "new_turn")
    assert any(item.get("type") == "compaction" for item in state["messages"]["items"])


@pytest.mark.asyncio
async def test_compaction_extension_async_path_compacts_once():
    model = _CompactingModel()
    messages = _history_with_completed_turn()
    agent = Agent(
        name="agent",
        model=model,
        extensions=[
            CompactionExtension(
                CompactionPolicy(
                    trigger_ratio=0.8,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            )
        ],
    )

    output = await agent.acall(
        "New question",
        messages=messages,
        scope=ExecutionScope(
            thread_id="thread_1",
            namespace="agent",
            run_id="new_turn",
        ),
    )

    assert output == "ok"
    assert len(model.compaction_calls) == 1


def test_compaction_extension_skips_when_threshold_is_not_reached():
    model = _CompactingModel()
    agent = Agent(
        name="agent",
        model=model,
        extensions=[
            CompactionExtension(
                CompactionPolicy(
                    trigger_ratio=1,
                    reserved_output_tokens=0,
                    safety_margin_tokens=0,
                )
            )
        ],
    )

    agent("New question", messages=_history_with_completed_turn())

    assert model.compaction_calls == []


def test_compaction_policy_is_an_encodable_immutable_struct():
    policy = CompactionPolicy(
        trigger_ratio=0.75,
        reserved_output_tokens=2048,
        safety_margin_tokens=512,
    )

    assert msgspec.json.decode(msgspec.json.encode(policy)) == {
        "trigger_ratio": 0.75,
        "reserved_output_tokens": 2048,
        "safety_margin_tokens": 512,
        "context_capacity": None,
    }
    with pytest.raises(AttributeError):
        policy.trigger_ratio = 0.5


def test_compaction_hook_without_capability_does_not_enable_compaction():
    model = _CompactingModel()

    def force_compaction(ctx):
        return replace(ctx, action="compact")

    agent = Agent(
        name="agent",
        model=model,
        hooks=[Hook(event="before_compaction", handler=force_compaction)],
    )

    agent("New question", messages=_history_with_completed_turn())

    assert model.compaction_calls == []


def test_extension_capability_follows_active_run_snapshot():
    agent = Agent(name="agent", model=_RecordingModel())
    handle = agent.register_extension("compaction", CompactionExtension())

    with agent._extension_run_snapshot():
        handle.remove()
        assert agent.has_extension_capability("context_compaction") is True

    assert agent.has_extension_capability("context_compaction") is False


@pytest.mark.parametrize(
    ("capabilities", "error", "match"),
    [
        ("context_compaction", TypeError, "not a single string"),
        (("duplicate", "duplicate"), ValueError, "duplicate capabilities"),
        (("",), ValueError, "non-empty strings"),
    ],
)
def test_extension_capabilities_are_validated(capabilities, error, match):
    class InvalidCapabilitiesExtension(AgentExtension):
        def __init__(self):
            super().__init__("invalid")

        def capabilities(self):
            return capabilities

    agent = Agent(name="agent", model=_RecordingModel())

    with pytest.raises(error, match=match):
        agent.register_extension("invalid", InvalidCapabilitiesExtension())

    assert not agent.has_extension("invalid")


def test_builtin_tool_feedback_extension_resolves_direct_output():
    agent = Agent(name="agent", model=_RecordingModel())
    intents, outcomes = _tool_feedback_values("direct")

    with execution_context(scope=ExecutionScope(thread_id="thread_1")):
        feedback = agent._resolve_tool_feedback(
            intents,
            outcomes,
            messages=ChatMessages(),
            vars={},
            reasoning="checked",
        )

    assert agent.has_extension("tool_feedback")
    assert feedback.action == "return"
    assert feedback.output.tool_responses.reasoning == "checked"
    assert feedback.output.tool_responses.tool_calls[0]["result"] == "result_0"


def test_builtin_tool_feedback_extension_rejects_mixed_return_modes():
    agent = Agent(name="agent", model=_RecordingModel())
    intents, outcomes = _tool_feedback_values("direct", "handoff")

    with (
        execution_context(scope=ExecutionScope(thread_id="thread_1")),
        pytest.raises(ValueError, match="incompatible return feedback"),
    ):
        agent._resolve_tool_feedback(
            intents,
            outcomes,
            messages=ChatMessages(),
            vars={},
            reasoning=None,
        )


@pytest.mark.asyncio
async def test_custom_tool_feedback_extension_can_resolve_new_mode():
    class ApprovalFeedbackExtension(AgentExtension):
        def __init__(self):
            super().__init__("tool_feedback")

        async def resolve(self, ctx: ToolFeedbackContext):
            await asyncio.sleep(0)
            if {outcome.feedback.name for outcome in ctx.outcomes} == {"approval"}:
                return replace(ctx, action="return", output="awaiting approval")
            return ctx

        def hooks(self):
            return (Hook(event="resolve_tool_feedback", handler=self.resolve),)

    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        extensions=[ApprovalFeedbackExtension()],
    )
    intents, outcomes = _tool_feedback_values("approval")

    with execution_context(scope=ExecutionScope(thread_id="thread_1")):
        feedback = await agent._aresolve_tool_feedback(
            intents,
            outcomes,
            messages=ChatMessages(),
            vars={},
            reasoning=None,
        )

    assert feedback.action == "return"
    assert feedback.output == "awaiting approval"


@pytest.mark.asyncio
async def test_custom_tool_feedback_extension_ends_real_tool_loop():
    class ApprovalFeedbackExtension(AgentExtension):
        def __init__(self):
            super().__init__("tool_feedback")

        async def resolve(self, ctx: ToolFeedbackContext):
            if {outcome.feedback.name for outcome in ctx.outcomes} == {"approval"}:
                return replace(ctx, action="return", output="awaiting approval")
            return ctx

        def hooks(self):
            return (Hook(event="resolve_tool_feedback", handler=self.resolve),)

    class ToolCallingModel:
        model_type = "chat_completion"

        def __init__(self):
            self.calls = 0

        async def acall(self, **_kwargs):
            self.calls += 1
            response = ModelResponse()
            response.set_response_type("tool_call")
            calls = ToolCallAggregator()
            calls.process(0, "call_1", "deploy", '{"environment":"staging"}')
            response.add(calls)
            response.metadata = {}
            return response

    @tool_config(feedback="approval")
    def deploy(environment: str) -> str:
        return f"deployment:{environment}"

    model = ToolCallingModel()
    agent = Agent(
        name="agent",
        model=model,
        tools=[deploy],
        extensions=[ApprovalFeedbackExtension()],
    )

    output = await agent.acall("Deploy staging")

    assert output == "awaiting approval"
    assert model.calls == 1


def test_extension_contributes_and_removes_hooks_and_tools():
    model = _RecordingModel()
    agent = Agent(name="agent", model=model)

    handle = agent.register_extension("prompt", _PromptExtension())

    assert handle.active
    assert "extension prompt" in agent.get_system_prompt()
    assert "extension_tool" in agent.tool_library.library

    handle.remove()

    assert not handle.active
    assert "extension prompt" not in agent.get_system_prompt()
    assert "extension_tool" not in agent.tool_library.library


@pytest.mark.asyncio
async def test_extension_removal_preserves_active_run_snapshot():
    started = asyncio.Event()
    release = asyncio.Event()
    model = _RecordingModel()

    class BlockingExtension(_PromptExtension):
        async def add_prompt(self, ctx: ModelContext):
            started.set()
            await release.wait()
            return replace(
                ctx,
                system_prompt=f"{ctx.system_prompt}\nactive snapshot".strip(),
            )

        def hooks(self):
            return (Hook(event="transform_system_prompt", handler=self.add_prompt),)

    agent = Agent(name="agent", model=model)
    handle = agent.register_extension("prompt", BlockingExtension())

    active_run = asyncio.create_task(agent.acall("first"))
    await started.wait()
    handle.remove()

    assert await agent.acall("second") == "ok"
    assert "active snapshot" not in (model.calls[0]["system_prompt"] or "")
    assert [tool.name for tool in model.calls[0]["tool_catalog"].tool_entries()] == [
        "extension_tool"
    ]

    release.set()
    assert await active_run == "ok"
    assert "active snapshot" in (model.calls[1]["system_prompt"] or "")
    assert "extension_tool" not in agent.tool_library.library


@pytest.mark.asyncio
async def test_async_extension_removal_waits_for_active_run_cleanup():
    started = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    class AsyncCleanupExtension(AgentExtension):
        def __init__(self):
            super().__init__("cleanup")

        async def block(self, ctx):
            started.set()
            await release.wait()
            return ctx

        def hooks(self):
            return (Hook(event="transform_system_prompt", handler=self.block),)

        async def aon_remove(self, agent):
            cleaned.set()

    agent = Agent(name="agent", model=_RecordingModel())
    handle = agent.register_extension("cleanup", AsyncCleanupExtension())
    run = asyncio.create_task(agent.acall("first"))
    await started.wait()

    removal = asyncio.create_task(handle.aremove())
    await asyncio.sleep(0)
    assert not cleaned.is_set()

    release.set()
    assert await run == "ok"
    await removal
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_async_extension_removal_wakes_safely_from_worker_thread():
    started = threading.Event()
    release = threading.Event()
    cleaned = asyncio.Event()

    class ThreadedCleanupExtension(AgentExtension):
        def __init__(self):
            super().__init__("threaded_cleanup")

        def block(self, ctx):
            started.set()
            release.wait(timeout=2)
            return ctx

        def hooks(self):
            return (Hook(event="transform_system_prompt", handler=self.block),)

        async def aon_remove(self, agent):
            cleaned.set()

    agent = Agent(name="agent", model=_RecordingModel())
    handle = agent.register_extension("threaded_cleanup", ThreadedCleanupExtension())
    run = asyncio.create_task(asyncio.to_thread(agent, "first"))
    assert await asyncio.to_thread(started.wait, 1)

    removal = asyncio.create_task(handle.aremove())
    await asyncio.sleep(0)
    assert not cleaned.is_set()

    release.set()
    assert await run == "ok"
    await asyncio.wait_for(removal, timeout=1)
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_abort_cancels_async_extension_hook():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class NetworkExtension(AgentExtension):
        def __init__(self):
            super().__init__("network")

        async def analyze(self, payload):
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()
            return payload

        def hooks(self):
            return (Hook(event="before_run", handler=self.analyze),)

    signal = AbortSignal()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        extensions=[NetworkExtension()],
    )
    run = asyncio.create_task(
        agent.acall("hello", scope=ExecutionScope(abort_signal=signal))
    )
    await started.wait()

    signal.abort("test cancellation")

    with pytest.raises(AbortRequestedError, match="test cancellation"):
        await run
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_abort_reaches_transform_output_after_agent_forward():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class OutputExtension(AgentExtension):
        def __init__(self):
            super().__init__("output")

        async def transform(self, output):
            started.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()
            return output

        def hooks(self):
            return (Hook(event="transform_output", handler=self.transform),)

    signal = AbortSignal()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        extensions=[OutputExtension()],
    )
    run = asyncio.create_task(
        agent.acall("hello", scope=ExecutionScope(abort_signal=signal))
    )
    await started.wait()

    signal.abort("stop output")

    with pytest.raises(AbortRequestedError, match="stop output"):
        await run
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_agent_output_context_exposes_runtime_vars_and_scope():
    signal = AbortSignal()
    scope = ExecutionScope(abort_signal=signal)

    def transform(ctx):
        assert ctx.vars == {"tenant": "acme"}
        assert ctx.scope.abort_signal is signal
        return replace(ctx, output=f"{ctx.output}:{ctx.vars['tenant']}")

    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        hooks=[Hook(event="transform_output", handler=transform)],
    )

    assert await agent.acall("hello", vars={"tenant": "acme"}, scope=scope) == "ok:acme"


@pytest.mark.asyncio
async def test_typed_model_lifecycle_contexts_transform_request_and_response():
    observed = []

    async def transform_conversation(ctx):
        assert isinstance(ctx, ConversationContext)
        transformed = ctx.messages.copy()
        transformed.add_system("extension context")
        return replace(ctx, messages=transformed)

    async def prepare_request(ctx):
        assert isinstance(ctx, ModelRequestContext)
        observed.append(ctx)
        return replace(ctx, system_prompt="extension request")

    async def transform_response(ctx):
        assert isinstance(ctx, ModelResponseContext)
        ctx.response.data = "transformed response"
        return ctx

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        hooks=[
            Hook(event="transform_context", handler=transform_conversation),
            Hook(event="before_request", handler=prepare_request),
            Hook(event="after_response", handler=transform_response),
        ],
    )

    assert await agent.acall(
        "hello", vars={"tenant": "acme"}, messages=ChatMessages()
    ) == ("transformed response")
    assert isinstance(observed[0], ModelRequestContext)
    assert observed[0].vars == {"tenant": "acme"}
    assert model.calls[0]["system_prompt"] == "extension request"
    assert any(
        item["role"] == "system" and item["content"] == "extension context"
        for item in model.calls[0]["messages"].to_chatml()
    )


def test_transform_tool_catalog_changes_current_request_surface():
    def alpha() -> str:
        """Return alpha."""
        return "alpha"

    def beta() -> str:
        """Return beta."""
        return "beta"

    def keep_alpha(ctx):
        assert isinstance(ctx, ToolCatalogContext)
        return replace(ctx, catalog=ctx.catalog.with_tools(("alpha",)))

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        tools=[alpha, beta],
        hooks=[Hook(event="transform_tool_catalog", handler=keep_alpha)],
    )

    assert agent("hello") == "ok"
    assert [tool.name for tool in model.calls[0]["tool_catalog"].tool_entries()] == [
        "alpha"
    ]


@pytest.mark.asyncio
async def test_async_transform_tool_catalog_preserves_canonical_entry_metadata():
    def alpha() -> str:
        """Return alpha."""
        return "alpha"

    async def select_alpha(ctx):
        entry = ctx.catalog.tool_entries()[0]
        assert entry.ref.library_id == ctx.catalog.library_id
        return replace(ctx, catalog=ctx.catalog.with_tools(("alpha",)))

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        tools=[alpha],
        hooks=[Hook(event="transform_tool_catalog", handler=select_alpha)],
    )

    assert await agent.acall("hello") == "ok"
    assert [tool.name for tool in model.calls[0]["tool_catalog"].tool_entries()] == [
        "alpha"
    ]


@pytest.mark.asyncio
async def test_async_transform_notifications_filters_model_context():
    seen = []

    async def suppress(ctx):
        assert isinstance(ctx, NotificationContext)
        seen.extend(ctx.notifications)
        return replace(ctx, notifications=())

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        hooks=[Hook(event="transform_notifications", handler=suppress)],
    )
    agent.agent_inbox.publish(
        {"source": "task", "status": "completed", "ref": "task_1"}
    )

    assert await agent.acall("continue", messages=ChatMessages()) == "ok"
    assert len(seen) == 1
    assert all(
        "<notification " not in str(item.get("content"))
        for item in model.calls[0]["messages"].to_chatml()
    )


def test_run_end_hooks_wrap_final_checkpoint():
    trace = []

    class RecordingStore(InMemoryCheckpointStore):
        def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
            trace.append("checkpoint")
            return super().commit_state(namespace, thread_id, run_id, state, **kwargs)

    def before(ctx):
        assert isinstance(ctx, RunEndContext)
        trace.append("before")
        return replace(ctx, output=f"{ctx.output}:before")

    def after(ctx):
        assert isinstance(ctx, RunEndContext)
        trace.append("after")
        return replace(ctx, output=f"{ctx.output}:after")

    history = ChatMessages()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        checkpoint_store=RecordingStore(),
        hooks=[
            Hook(event="before_run_end", handler=before),
            Hook(event="after_run_end", handler=after),
        ],
    )

    assert agent("hello", messages=history) == "ok:before:after"
    assert trace == ["before", "checkpoint", "after"]


def test_after_run_end_failure_preserves_committed_run_and_output():
    trace = []

    class RecordingStore(InMemoryCheckpointStore):
        def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
            trace.append(("checkpoint", state["status"]))
            return super().commit_state(namespace, thread_id, run_id, state, **kwargs)

    def fail_after(ctx):
        trace.append(("after", ctx.outcome))
        raise RuntimeError("observer failed")

    store = RecordingStore()
    history = ChatMessages()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        checkpoint_store=store,
        hooks=[Hook(event="after_run_end", handler=fail_after)],
    )
    scope = ExecutionScope(thread_id="thread", namespace="agent", run_id="run")

    with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
        result = agent("hello", messages=history, scope=scope)

    assert result == "ok"
    assert trace == [("checkpoint", "completed"), ("after", "completed")]
    assert store.load_state("agent", "thread", "run")["status"] == "completed"
    assert history.turns[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_async_after_run_end_failure_preserves_committed_run_and_output():
    calls = []

    async def fail_after(ctx):
        calls.append(ctx.outcome)
        raise RuntimeError("observer failed")

    store = InMemoryCheckpointStore()
    history = ChatMessages()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        checkpoint_store=store,
        hooks=[Hook(event="after_run_end", handler=fail_after)],
    )
    scope = ExecutionScope(
        thread_id="async_thread",
        namespace="agent",
        run_id="async_run",
    )

    with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
        result = await agent.acall("hello", messages=history, scope=scope)

    assert result == "ok"
    assert calls == ["completed"]
    assert store.load_state("agent", "async_thread", "async_run")["status"] == (
        "completed"
    )
    assert history.turns[-1]["status"] == "completed"


def test_run_end_hooks_receive_failed_outcome_around_checkpoint():
    trace = []

    class FailingModel:
        model_type = "chat_completion"

        def __call__(self, **_kwargs):
            raise RuntimeError("model failed")

    class RecordingStore(InMemoryCheckpointStore):
        def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
            trace.append(("checkpoint", state["status"]))
            return super().commit_state(namespace, thread_id, run_id, state, **kwargs)

    def record_before(ctx):
        trace.append(("before", ctx.outcome))
        return ctx

    def record_after(ctx):
        trace.append(("after", ctx.outcome))
        return ctx

    agent = Agent(
        name="agent",
        model=FailingModel(),
        checkpoint_store=RecordingStore(),
        hooks=[
            Hook(event="before_run_end", handler=record_before),
            Hook(event="after_run_end", handler=record_after),
        ],
    )

    with pytest.raises(RuntimeError, match="model failed"):
        agent("hello", messages=ChatMessages())

    assert trace == [
        ("before", "failed"),
        ("checkpoint", "failed"),
        ("after", "failed"),
    ]


def test_before_run_end_failure_is_settled_without_repeating_hook():
    calls = []
    store = InMemoryCheckpointStore()

    def fail_before(ctx):
        calls.append(("before", ctx.outcome))
        raise RuntimeError("cannot finalize")

    def record_after(ctx):
        calls.append(("after", ctx.outcome))
        return ctx

    history = ChatMessages()
    agent = Agent(
        name="agent",
        model=_RecordingModel(),
        checkpoint_store=store,
        hooks=[
            Hook(event="before_run_end", handler=fail_before),
            Hook(event="after_run_end", handler=record_after),
        ],
    )
    scope = ExecutionScope(
        thread_id="before_failure_thread",
        namespace="agent",
        run_id="before_failure_run",
    )

    with pytest.raises(RuntimeError, match="cannot finalize"):
        agent("hello", messages=history, scope=scope)

    assert calls == [("before", "completed"), ("after", "failed")]
    assert (
        store.load_state("agent", "before_failure_thread", "before_failure_run")[
            "status"
        ]
        == "failed"
    )
    assert history.turns[-1]["status"] == "failed"


def test_after_run_end_failure_cannot_replace_original_run_error():
    class FailingModel:
        model_type = "chat_completion"

        def __call__(self, **_kwargs):
            raise ValueError("model failed")

    def fail_after(_ctx):
        raise RuntimeError("observer failed")

    store = InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=FailingModel(),
        checkpoint_store=store,
        hooks=[Hook(event="after_run_end", handler=fail_after)],
    )
    scope = ExecutionScope(
        thread_id="failed_thread",
        namespace="agent",
        run_id="failed_run",
    )

    with (
        pytest.warns(RuntimeWarning, match="after_run_end hook failed"),
        pytest.raises(ValueError, match="model failed"),
    ):
        agent("hello", messages=ChatMessages(), scope=scope)

    assert store.load_state("agent", "failed_thread", "failed_run")["status"] == (
        "failed"
    )


@pytest.mark.asyncio
async def test_extension_state_is_shared_within_run_and_isolated_between_runs():
    class StatefulExtension(AgentExtension):
        def __init__(self):
            super().__init__("stateful")

        def remember(self, payload):
            self.state()["message"] = payload.message
            return payload

        def add_prompt(self, ctx: ModelContext):
            message = self.state()["message"]
            return replace(
                ctx,
                system_prompt=f"{ctx.system_prompt}\nrun:{message}".strip(),
            )

        def hooks(self):
            return (
                Hook(event="before_run", handler=self.remember),
                Hook(event="transform_system_prompt", handler=self.add_prompt),
            )

    model = _RecordingModel()
    agent = Agent(
        name="agent",
        model=model,
        extensions=[StatefulExtension()],
    )

    assert await asyncio.gather(agent.acall("one"), agent.acall("two")) == [
        "ok",
        "ok",
    ]
    assert {call["system_prompt"] for call in model.calls} == {"run:one", "run:two"}


@pytest.mark.asyncio
async def test_abort_cancels_foreground_async_tool():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool() -> str:
        """Wait until the caller cancels this tool."""
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return "late"

    signal = AbortSignal()
    library = ToolLibrary("agent", [slow_tool])
    with execution_context(scope=ExecutionScope(abort_signal=signal)):
        run = asyncio.create_task(library.acall([("call_1", "slow_tool", {})]))
        await started.wait()
        signal.abort("stop tool")
        with pytest.raises(AbortRequestedError, match="stop tool"):
            await run

    assert cancelled.is_set()
