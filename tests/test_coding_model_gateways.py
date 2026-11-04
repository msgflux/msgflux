"""Integration tests for coding model gateways (factory + selection policy).

Scripted ``BaseModel`` subclasses stand in for real providers: no network,
no credentials. They exercise the real ``ModelGateway``, factory, and
``CodingModelSelection`` hooks rather than mocking the gateway away.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from msgflux.coding.checkpoints import CodingCheckpointExtension
from msgflux.coding.config import CodingAgentCatalog, coding_agent_options
from msgflux.coding.factory import create_coding_session_factory
from msgflux.coding.model_selection import CodingGatewayReadTool, CodingModelSelection
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import ModelRouterError, TaskPauseRequestedError
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.hooks import ConversationContext
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.service import ServiceThread


# ---------------------------------------------------------------------------
# Scripted BaseModel
# ---------------------------------------------------------------------------


def _text(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool_call(name, arguments, call_id="call_1"):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, json.dumps(arguments))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


class ScriptedModel(BaseModel):
    """A real BaseModel whose responses are scripted per instance."""

    model_type = "chat_completion"

    def __init__(
        self,
        model_id,
        *,
        provider="other",
        script=None,
        vision=False,
        barrier=None,
    ):
        self.model_id = model_id
        self.provider = provider
        self.profile = SimpleNamespace(
            modalities=SimpleNamespace(input=("image", "text") if vision else ("text",))
        )
        self.script = list(script or [])
        self.calls = []
        self.closed = False
        self.credential_resolver = None
        self.reasoning_effort = None
        self.barrier = barrier

    def _initialize(self):
        pass

    def close(self):
        self.closed = True

    def set_reasoning_effort(self, reasoning_effort):
        self.reasoning_effort = reasoning_effort
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        action = self.script.pop(0) if self.script else _text(f"from {self.model_id}")
        if isinstance(action, BaseException):
            raise action
        if action == "lookup-tool":
            return _tool_call("lookup", {"query": "q"})
        return action

    async def acall(self, **kwargs):
        return self.__call__(**kwargs)


def _gateway_options(*, fallback=False, subagents=(), accounts=("fast", "deep")):
    deployments = {
        "fast": {
            "model_name": "fast",
            "model": "openai-codex/fast-model",
            "description": "Fast model",
        },
        "deep": {
            "model_name": "deep",
            "model": "anthropic/deep-model",
            "description": "Deep model",
            "reasoning_effort": "high",
            "account_name": "work",
        },
    }
    from msgflux.coding.config import CodingModelDeployment

    return coding_agent_options(
        "coding",
        models=tuple(CodingModelDeployment(**deployments[name]) for name in accounts),
        model_fallback=fallback,
        subagents=subagents,
    )


def _scripted_pair(
    *,
    fast_script=None,
    deep_script=None,
    fast_vision=False,
    deep_vision=False,
    barrier=None,
    fallback=False,
):
    fast = ScriptedModel(
        "fast-model",
        provider="openai-codex",
        script=fast_script,
        vision=fast_vision,
        barrier=barrier,
    )
    deep = ScriptedModel(
        "deep-model",
        provider="anthropic",
        script=deep_script,
        vision=deep_vision,
        barrier=barrier,
    )
    gateway = ModelGateway(
        [
            {"model_name": "fast", "model": fast, "description": "Fast model"},
            {"model_name": "deep", "model": deep, "description": "Deep model"},
        ],
        fallback=fallback,
    )
    return gateway, fast, deep


def _lookup_tool(calls):
    def lookup(query: str) -> str:
        """Look up one entry."""
        calls.append(query)
        return "found"

    return lookup


def _scope(run_id="run", account_name="default"):
    return ExecutionScope(
        thread_id="thread",
        namespace="coding",
        run_id=run_id,
        account_name=account_name,
    )


def _agent(gateway, options, store, calls, selection=None):
    from msgflux.coding.factory import _tool_instances

    selection = selection or CodingModelSelection(options, gateway.models)
    # Register the real builtin workspace tools (union across deployments),
    # adapted exactly as the factory adapts them for gateway agents.
    builtin = selection.adapt_tools(
        _tool_instances(
            options.tools_active,
            options.tools_deferred,
            owned_resources=[],
            supports_vision=False,
        )
    )
    agent = Agent(
        name="coding",
        model=gateway,
        tools=[*builtin, _lookup_tool(calls)],
        checkpoint_store=store,
        extensions=[selection, CodingCheckpointExtension()],
    )
    return agent, selection


def _binding(store, run_id="run"):
    state = store.load_state("coding", "thread", run_id)
    assert state is not None
    return state["runtime"]["extensions"]["coding_models"]["binding"]


# ---------------------------------------------------------------------------
# Factory: lazy gateway construction, accounts, child pools, cleanup
# ---------------------------------------------------------------------------


class _FakeAccounts:
    def __init__(self):
        self.calls = []

    def resolver(self, provider, account_name=None, allow_account_override=False):
        self.calls.append((provider, account_name, allow_account_override))
        return SimpleNamespace(
            provider=provider,
            account_name=account_name,
            allow_account_override=allow_account_override,
        )


def _patch_chat_completion(monkeypatch, scripts_by_path, created, failures=()):
    from msgflux.models.model import Model

    def fake_chat_completion(model_path, **kwargs):
        if model_path in failures:
            raise RuntimeError(f"construction failed for {model_path}")
        provider, model_id = model_path.split("/", 1)
        model = ScriptedModel(
            model_id,
            provider=provider,
            script=scripts_by_path.get(model_path),
            vision=model_id == "vision-model",
        )
        model.credential_resolver = kwargs.get("credential_resolver")
        model.reasoning_effort = kwargs.get("reasoning_effort")
        created.append(model)
        return model

    monkeypatch.setattr(Model, "chat_completion", staticmethod(fake_chat_completion))


def _make_factory_session(
    tmp_path,
    monkeypatch,
    options,
    *,
    catalog=None,
    accounts=None,
    created=None,
    thread_model=None,
    failures=(),
):
    created = created if created is not None else []
    _patch_chat_completion(monkeypatch, {}, created, failures=failures)
    factory = create_coding_session_factory(
        options=options,
        catalog=catalog or CodingAgentCatalog({"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
        accounts=accounts,
    )
    session = asyncio.run(
        factory(
            ServiceThread(
                "thread-1",
                "coding",
                str(tmp_path),
                interactive=False,
                model=thread_model,
            )
        )
    )
    return session, created


def test_factory_lazily_builds_child_gateway_with_descriptions(tmp_path, monkeypatch):
    options = _gateway_options()
    created = []
    _patch_chat_completion(monkeypatch, {}, created)
    factory = create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog({"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
        accounts=_FakeAccounts(),
    )
    assert created == []  # lazy: nothing constructed until a session is opened

    session = asyncio.run(
        factory(ServiceThread("thread-1", "coding", str(tmp_path), interactive=False))
    )
    try:
        agent = session.agent
        gateway = agent.model
        assert isinstance(gateway, ModelGateway)
        assert gateway.fallback is False
        assert gateway.model_names == ["fast", "deep"]
        assert gateway.get_model_description("fast") == "Fast model"
        assert gateway.get_model_description("deep") == "Deep model"
        assert [m.provider for m in gateway.models] == ["openai-codex", "anthropic"]
        assert [m.model_id for m in gateway.models] == ["fast-model", "deep-model"]
        # Per-deployment reasoning effort reaches construction.
        assert gateway.models[0].reasoning_effort == "medium"
        assert gateway.models[1].reasoning_effort == "high"
        # Selection extension is registered before the checkpoint extension.
        assert agent.has_extension("coding_models")
        names = list(agent.extensions)
        assert names.index("coding_models") < names.index("coding_checkpoints")
    finally:
        asyncio.run(session.on_close())
    assert all(m.closed for m in created)


def test_factory_uses_provider_resolvers_explicit_and_inherited_accounts(
    tmp_path, monkeypatch
):
    options = _gateway_options()
    accounts = _FakeAccounts()
    session, created = _make_factory_session(
        tmp_path, monkeypatch, options, accounts=accounts
    )
    try:
        # fast omits account_name -> plain provider resolver (inherits execution
        # account); deep pins "work" -> explicit override resolver.
        assert accounts.calls == [
            ("openai-codex", None, False),
            ("anthropic", "work", True),
        ]
        fast, deep = session.agent.model.models
        assert fast.credential_resolver.account_name is None
        assert fast.credential_resolver.allow_account_override is False
        assert deep.credential_resolver.account_name == "work"
        assert deep.credential_resolver.allow_account_override is True
    finally:
        asyncio.run(session.on_close())
    assert all(m.closed for m in created)


def test_factory_child_pool_and_cli_override_preserve_children(tmp_path, monkeypatch):
    parent = _gateway_options(subagents=("child",))
    child = coding_agent_options("child", "other/child-model")
    catalog = CodingAgentCatalog({"coding": parent, "child": child})
    session, created = _make_factory_session(
        tmp_path, monkeypatch, parent, catalog=catalog, thread_model="provider/override"
    )
    try:
        root = session.agent
        # CLI --model override replaces the root gateway with a single model.
        assert not isinstance(root.model, ModelGateway)
        assert root.model.provider == "provider" and root.model.model_id == "override"
        assert not root.has_extension("coding_models")
        child_agent = root.tool_library.get_tool_definition("child").executor.impl
        assert child_agent.model.provider == "other"
        assert child_agent.model.model_id == "child-model"
        assert {m.model_id for m in created} == {"override", "child-model"}
    finally:
        asyncio.run(session.on_close())
    assert all(m.closed for m in created)


def test_factory_partial_construction_closes_first_deployment(tmp_path, monkeypatch):
    options = _gateway_options()
    created = []
    _patch_chat_completion(monkeypatch, {}, created, failures={"anthropic/deep-model"})
    factory = create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog({"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
        accounts=_FakeAccounts(),
    )
    with pytest.raises(RuntimeError, match="construction failed"):
        asyncio.run(
            factory(
                ServiceThread("thread-1", "coding", str(tmp_path), interactive=False)
            )
        )
    assert [m.model_id for m in created] == ["fast-model"]
    assert created[0].closed is True


def test_factory_rejects_deployment_account_without_managed_accounts(
    tmp_path, monkeypatch
):
    # Production guard: an explicit account_name with no managed Accounts
    # must fail rather than silently fall back to environment credentials.
    options = _gateway_options()
    created = []
    _patch_chat_completion(monkeypatch, {}, created)
    factory = create_coding_session_factory(
        options=options,
        catalog=CodingAgentCatalog({"coding": options}),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "state",
    )
    with pytest.raises(ValueError, match="requires managed Accounts"):
        asyncio.run(
            factory(
                ServiceThread("thread-1", "coding", str(tmp_path), interactive=False)
            )
        )
    # The already-constructed first deployment is still cleaned up.
    assert [m.model_id for m in created] == ["fast-model"]
    assert created[0].closed is True


# ---------------------------------------------------------------------------
# Selection policy: defaults, unknown alias, fallback=False, tool projection
# ---------------------------------------------------------------------------


def test_first_alias_is_default_without_model_preference():
    gateway, fast, deep = _scripted_pair()
    options = _gateway_options()
    store, calls = InMemoryCheckpointStore(), []
    agent, _ = _agent(gateway, options, store, calls)

    assert agent("hello", scope=_scope(), model_preference=None) == "from fast-model"
    assert len(fast.calls) == 1 and deep.calls == []
    binding = _binding(store)
    assert binding["model_name"] == "fast"
    # Omitted account_name is durably recorded as the inherited scope account.
    assert binding["models"][0]["account_name"] == "default"
    # No shared gateway state was mutated by the request.
    assert gateway.current_model_index == 0


def test_unknown_alias_rejects_before_any_model_call():
    gateway, fast, deep = _scripted_pair()
    options = _gateway_options()
    agent, _ = _agent(gateway, options, InMemoryCheckpointStore(), [])

    with pytest.raises(ValueError, match=r"Unknown (coding model alias|model)"):
        agent("hello", scope=_scope(), model_preference="missing")
    assert fast.calls == [] and deep.calls == []


def test_fallback_false_does_not_call_second_on_failure():
    gateway, fast, deep = _scripted_pair(fast_script=[RuntimeError("fast exploded")])
    options = _gateway_options()
    agent, _ = _agent(gateway, options, InMemoryCheckpointStore(), [])

    with pytest.raises(ModelRouterError):
        agent("hello", scope=_scope(), model_preference="fast")
    assert len(fast.calls) == 1 and deep.calls == []


@pytest.mark.parametrize(
    "preference, present, absent",
    [
        ("fast", {"apply_patch"}, {"edit", "write", "delete"}),
        ("deep", {"edit", "write", "delete"}, {"apply_patch"}),
    ],
)
def test_selected_model_tool_catalog_projection(preference, present, absent):
    gateway, fast, deep = _scripted_pair()
    options = _gateway_options()
    agent, selection = _agent(gateway, options, InMemoryCheckpointStore(), [])

    agent("hello", scope=_scope(), model_preference=preference)
    model = {"fast": fast, "deep": deep}[preference]
    catalog = model.calls[0]["tool_catalog"]
    names = {entry.name for entry in catalog.visible_entries()}
    # The shared library still holds the union; only the request catalog is
    # projected for the selected provider family.
    assert present <= names
    assert not (absent & names)
    assert "bash" in names and "read" in names
    assert selection.tool_names >= present | {"bash", "read"}


def test_concurrent_runs_keep_alias_toolsets_and_shared_state():
    barrier = threading.Barrier(2)
    gateway, fast, deep = _scripted_pair(barrier=barrier)
    options = _gateway_options()
    store, calls = InMemoryCheckpointStore(), []
    agent, _ = _agent(gateway, options, store, calls)
    shared_models = list(gateway.models)
    read_tool = agent.tool_library.library["read"].impl

    def run(preference):
        return agent(
            "hello",
            scope=_scope(run_id=f"run-{preference}"),
            model_preference=preference,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(
            zip(
                ["fast", "deep"],
                pool.map(run, ["fast", "deep"]),
            )
        )

    assert results == {"fast": "from fast-model", "deep": "from deep-model"}
    fast_names = {e.name for e in fast.calls[0]["tool_catalog"].visible_entries()}
    deep_names = {e.name for e in deep.calls[0]["tool_catalog"].visible_entries()}
    assert "apply_patch" in fast_names and "edit" not in fast_names
    assert "edit" in deep_names and "apply_patch" not in deep_names
    # Shared gateway/tool instances were not mutated by either run.
    assert gateway.models == shared_models
    assert gateway.current_model_index == 0
    assert agent.tool_library.library["read"].impl is read_tool


# ---------------------------------------------------------------------------
# Durable pause / resume binding
# ---------------------------------------------------------------------------


def _paused_agent():
    gateway, fast, deep = _scripted_pair(
        deep_script=[
            "lookup-tool",
            TaskPauseRequestedError("run", "pause after tool"),
            _text("done-deep"),
        ]
    )
    options = _gateway_options()
    store, calls = InMemoryCheckpointStore(), []
    agent, _ = _agent(gateway, options, store, calls)
    return agent, gateway, fast, deep, options, store, calls


@pytest.mark.parametrize("repeat_preference", [True, False])
def test_pause_after_tool_resumes_same_alias_and_account(repeat_preference):
    agent, gateway, fast, deep, options, store, calls = _paused_agent()

    with pytest.raises(TaskPauseRequestedError):
        agent("go", scope=_scope(), model_preference="deep")
    assert calls == ["q"]  # tool ran once, before the pause
    assert len(deep.calls) == 2 and fast.calls == []
    binding = _binding(store)
    assert binding["model_name"] == "deep"
    assert binding["fallback"] is False
    # Without fallback, only the selected deployment is pinned in durable state.
    assert [entry["model_name"] for entry in binding["models"]] == ["deep"]
    assert binding["models"][0]["account_name"] == "work"  # explicit override
    assert binding["models"][0]["model"] == "anthropic/deep-model"

    resumed, _ = _agent(gateway, options, store, calls)
    resume_args = {"model_preference": "deep"} if repeat_preference else {}
    assert resumed(scope=_scope(), **resume_args) == "done-deep"
    assert calls == ["q"]  # pending tool effect was not replayed
    assert len(deep.calls) == 3 and fast.calls == []


@pytest.mark.parametrize("mutate", ["path", "account"])
def test_changed_config_rejects_resume_before_model_call(mutate):
    agent, gateway, _fast, deep, _options, store, calls = _paused_agent()
    with pytest.raises(TaskPauseRequestedError):
        agent("go", scope=_scope(), model_preference="deep")
    calls_before = len(deep.calls)

    from msgflux.coding.config import CodingModelDeployment

    changed = coding_agent_options(
        "coding",
        models=(
            CodingModelDeployment(
                model_name="fast",
                model="openai-codex/fast-model",
                description="Fast model",
            ),
            CodingModelDeployment(
                model_name="deep",
                model="anthropic/other-deep"
                if mutate == "path"
                else "anthropic/deep-model",
                description="Deep model",
                account_name="work" if mutate == "path" else "other-account",
            ),
        ),
    )
    resumed, _ = _agent(gateway, changed, store, calls)
    scope = _scope()

    with pytest.raises(ValueError, match="changed for this run"):
        resumed(scope=scope, model_preference="deep")
    assert len(deep.calls) == calls_before  # rejected before any model call


def test_new_run_can_use_changed_config():
    agent, gateway, _fast, _deep, _options, store, calls = _paused_agent()
    with pytest.raises(TaskPauseRequestedError):
        agent("go", scope=_scope(), model_preference="deep")

    from msgflux.coding.config import CodingModelDeployment

    changed = coding_agent_options(
        "coding",
        models=(
            CodingModelDeployment(
                model_name="fast",
                model="openai-codex/fast-model",
                description="Fast model",
            ),
            CodingModelDeployment(
                model_name="deep",
                model="anthropic/other-deep",
                description="Deep model",
            ),
        ),
    )
    fresh, _ = _agent(gateway, changed, store, calls)
    assert fresh(scope=_scope(run_id="fresh"), model_preference="deep") == "done-deep"
    pinned = {
        entry["model_name"]: entry
        for entry in _binding(store, run_id="fresh")["models"]
    }
    assert pinned["deep"]["model"] == "anthropic/other-deep"


def test_unused_alias_and_description_changes_do_not_invalidate_pending_run():
    # Binding pins only the selected deployment (fallback=False), so changes
    # to an unused alias or to descriptions must not block resume.
    agent, gateway, _fast, _deep, _options, store, calls = _paused_agent()
    with pytest.raises(TaskPauseRequestedError):
        agent("go", scope=_scope(), model_preference="deep")

    from msgflux.coding.config import CodingModelDeployment

    changed = coding_agent_options(
        "coding",
        models=(
            CodingModelDeployment(
                model_name="fast",
                model="openai-codex/changed-fast",
                description="Changed fast description",
            ),
            CodingModelDeployment(
                model_name="deep",
                model="anthropic/deep-model",
                description="Changed deep description",
                reasoning_effort="high",
                account_name="work",
            ),
        ),
    )
    resumed, _ = _agent(gateway, changed, store, calls)
    assert resumed(scope=_scope(), model_preference="deep") == "done-deep"
    assert calls == ["q"]


def test_fallback_true_uses_union_catalog_and_conservative_vision():
    gateway, fast, deep = _scripted_pair(
        fast_script=[RuntimeError("fast exploded")],
        fast_vision=True,
        fallback=True,
    )
    options = _gateway_options(fallback=True)
    agent, selection = _agent(gateway, options, InMemoryCheckpointStore(), [])

    # Gateway fallback: fast fails, deep serves the request.
    assert agent("hello", scope=_scope(), model_preference="fast") == "from deep-model"
    assert len(fast.calls) == 1 and len(deep.calls) == 1
    catalog = fast.calls[0]["tool_catalog"]
    names = {entry.name for entry in catalog.visible_entries()}
    # Union catalog: both provider families are visible with fallback on.
    assert {"apply_patch", "edit", "write", "delete"} <= names
    # Conservative vision: deep cannot read images, so the projected read
    # guidance must not promise image attachment.
    read_entry = next(e for e in catalog.entries if e.name == "read")
    assert "Images read by this tool" not in read_entry.usage_guidance
    assert selection.reader().supports_vision is False
    # Fallback binding pins every deployment.
    binding = _binding(agent.checkpoint_store)
    assert [e["model_name"] for e in binding["models"]] == ["fast", "deep"]
    assert binding["fallback"] is True


def test_resume_without_binding_is_rejected():
    gateway, fast, deep = _scripted_pair()
    options = _gateway_options()
    store, calls = InMemoryCheckpointStore(), []
    agent, _ = _agent(gateway, options, store, calls)

    from msgflux.chat_messages import ChatMessages

    chat = ChatMessages(thread_id="thread", namespace="coding")
    chat.begin_turn(turn_id="run")
    chat.add_user("legacy checkpoint without selection state")
    store.save_state(
        "coding", "thread", "run", {"status": "running", "messages": chat._to_state()}
    )

    with pytest.raises(ValueError, match="no coding model binding"):
        agent(scope=_scope(), model_preference="deep")
    assert fast.calls == [] and deep.calls == []


# ---------------------------------------------------------------------------
# Per-model read vision
# ---------------------------------------------------------------------------


def test_gateway_read_tool_delegates_vision_per_bound_alias():
    gateway, _fast, _deep = _scripted_pair(fast_vision=True)
    options = _gateway_options()
    _agent_obj, selection = _agent(gateway, options, InMemoryCheckpointStore(), [])
    tool = CodingGatewayReadTool(selection)

    scope = _scope()
    # Each alias binding is pinned per run, so exercise each in its own run.
    with agent_run_context(
        AgentRun(namespace="coding", thread_id="thread", run_id="r1")
    ):
        selection.bind(
            ConversationContext(
                scope=scope, vars={}, messages=[], model_preference="fast"
            )
        )
        # Vision-capable fast deployment permits image reads.
        assert tool._read_options("pic.png", None, None)[2] is True
    with agent_run_context(
        AgentRun(namespace="coding", thread_id="thread", run_id="r2")
    ):
        selection.bind(
            ConversationContext(
                scope=scope, vars={}, messages=[], model_preference="deep"
            )
        )
        # The same shared tool now reflects the non-vision deployment.
        with pytest.raises(ValueError, match="Image reading is disabled"):
            tool._read_options("pic.png", None, None)
        # Text reads still delegate fine.
        assert tool._read_options("notes.txt", None, None)[2] is False


def test_coding_read_without_pinned_binding_fails_clearly():
    gateway, _, _ = _scripted_pair()
    _, selection = _agent(gateway, _gateway_options(), InMemoryCheckpointStore(), [])
    tool = CodingGatewayReadTool(selection)
    with agent_run_context(
        AgentRun(namespace="coding", thread_id="thread", run_id="unbound")
    ):
        with pytest.raises(ValueError, match="pinned model binding"):
            tool._read_options("file.txt", None, None)


def test_coding_read_preserves_original_byte_budget():
    from msgflux.tools.builtin import ReadFileTool

    gateway, _, _ = _scripted_pair()
    _, selection = _agent(gateway, _gateway_options(), InMemoryCheckpointStore(), [])
    original = ReadFileTool(max_text_bytes=8)
    tool = CodingGatewayReadTool(selection, original=original)
    with agent_run_context(
        AgentRun(namespace="coding", thread_id="thread", run_id="bounded")
    ):
        selection.bind(
            ConversationContext(scope=_scope(), messages=[], model_preference="deep")
        )
        with pytest.raises(ValueError, match="byte limit"):
            tool._result("file.txt", b"123456789", None)


def test_concurrent_workspace_reads_use_their_run_binding(tmp_path):
    import msgflux as mf
    from msgflux.tools.builtin import ReadFileTool

    (tmp_path / "README.md").write_text("Project marker copper-fern\n")
    read_call = lambda: _tool_call(
        "read", {"path": "README.md", "offset": 1, "limit": 20}
    )
    gateway, fast, deep = _scripted_pair(
        fast_script=[read_call(), _text("fast read")],
        deep_script=[read_call(), _text("deep read")],
        barrier=threading.Barrier(2),
    )
    options = _gateway_options()
    selection = CodingModelSelection(options, gateway.models)
    agent = Agent(
        name="coding",
        model=gateway,
        workspace=mf.AgentWorkspace.local(tmp_path),
        tools=selection.adapt_tools([ReadFileTool()]),
        checkpoint_store=InMemoryCheckpointStore(),
        extensions=[selection, CodingCheckpointExtension()],
    )

    def execute(alias):
        return agent("Read README", scope=_scope(run_id=alias), model_preference=alias)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(execute, ["fast", "deep"])) == [
                "fast read",
                "deep read",
            ]
        for model in (fast, deep):
            assert "copper-fern" in str(list(model.calls[1]["messages"]))
    finally:
        asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_service_pool_default_does_not_create_a_scalar_thread_override(tmp_path):
    from msgflux.runtime.service import (
        AgentConfiguration,
        AgentModelDeployment,
        AgentService,
        SQLiteServiceStore,
    )

    store = SQLiteServiceStore(str(tmp_path / "service.sqlite3"))
    service = AgentService(store=store, default_agent="coding")

    def no_session(thread):
        raise AssertionError("Opening a thread must stay lazy")

    service.register(
        "coding",
        no_session,
        configuration=AgentConfiguration(
            model="openai-codex/fast-model",
            supports_model_override=True,
            models=(
                AgentModelDeployment(
                    "fast", "openai-codex/fast-model", "Fast", is_default=True
                ),
            ),
        ),
    )
    try:
        default = await service.open_thread(thread_id="pool")
        explicit = await service.open_thread(
            thread_id="override", model="openai-codex/fast-model"
        )
        assert default.model is None
        assert explicit.model == "openai-codex/fast-model"
        assert store.thread("pool").model is None
        assert (await service.open_thread(thread_id="pool")).model is None
    finally:
        await service.aclose()
        store.close()


@pytest.mark.parametrize(
    "model_path", ["openai-codex/fast-model", "anthropic/deep-model", "other/override"]
)
def test_explicit_root_override_even_when_equal_to_pool_default(
    tmp_path, monkeypatch, model_path
):
    session, created = _make_factory_session(
        tmp_path,
        monkeypatch,
        _gateway_options(),
        accounts=_FakeAccounts(),
        thread_model=model_path,
    )
    try:
        assert not isinstance(session.agent.model, ModelGateway)
        assert session.agent.model.model_id == model_path.split("/", 1)[1]
        assert len(created) == 1
    finally:
        asyncio.run(session.agent.aclose())
        asyncio.run(session.on_close())


def test_factory_resume_rejects_changed_default_deployment(tmp_path, monkeypatch):
    import msgspec

    options = _gateway_options()
    session, _ = _make_factory_session(
        tmp_path, monkeypatch, options, accounts=_FakeAccounts()
    )
    session.agent.config["stream"] = False
    session.agent.model.models[0].script = [
        TaskPauseRequestedError("run", "simulate process exit boundary")
    ]
    scope = _scope().with_overrides(thread_id="thread-1")
    try:
        with pytest.raises(TaskPauseRequestedError):
            session.agent("start", scope=scope)
    finally:
        asyncio.run(session.agent.aclose())
        asyncio.run(session.on_close())

    changed = coding_agent_options(
        "coding",
        models=(
            msgspec.structs.replace(
                options.models[0], model="openai-codex/changed-fast"
            ),
            options.models[1],
        ),
    )
    reopened, created = _make_factory_session(
        tmp_path, monkeypatch, changed, accounts=_FakeAccounts()
    )
    reopened.agent.config["stream"] = False
    try:
        assert isinstance(reopened.agent.model, ModelGateway)
        with pytest.raises(ValueError, match="configuration changed"):
            reopened.agent(scope=scope)
        assert all(model.calls == [] for model in created)
    finally:
        asyncio.run(reopened.agent.aclose())
        asyncio.run(reopened.on_close())
