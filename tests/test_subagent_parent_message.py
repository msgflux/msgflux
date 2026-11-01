"""Messages composed with ordinary tools and scoped library handles."""

import asyncio
import json
from copy import deepcopy

import msgflux as mf
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.tools.builtin import AgentTool, SendParentMessageTool


def _text(text="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _call(name, params, call_id="send"):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, json.dumps(params))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


class _Model(BaseModel):
    def __init__(self, *responses):
        self.model_type = "chat_completion"
        self.model_id = "scripted"
        self.provider = "mock"
        self.responses = list(responses)
        self.histories = []

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        messages = kwargs["messages"]
        self.histories.append(
            deepcopy(
                messages.to_items()
                if isinstance(messages, ChatMessages)
                else list(messages)
            )
        )
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def _report(text="Found cause", call_id="send"):
    return _call("send_parent_message", {"message": text}, call_id)


def _inbox(store=None):
    return AgentInbox(
        store=store or InMemoryAgentInboxStore(),
        namespace="parent",
        thread_id="thread",
        run_id="parent_run",
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_scoped_handle_uses_effective_store_and_library_namespace(persistent, tmp_path):
    store = (
        SQLiteAgentInboxStore(tmp_path / "inbox.sqlite3")
        if persistent
        else InMemoryAgentInboxStore()
    )
    current = AgentInbox(
        store=store, namespace="child", thread_id="thread", run_id="child_run"
    )
    library = ToolLibrary(name="parent", tools=[])
    handle = library.get_handle()
    default = handle.get_agent_inbox()
    with execution_context(agent_inbox=current):
        assert handle.get_agent_inbox() is default
        target = handle.get_agent_inbox(thread_id="thread", run_id="parent_run")
    assert target.store is store
    assert (target.namespace, target.thread_id, target.run_id) == (
        default.namespace,
        "thread",
        "parent_run",
    )
    assert (current.namespace, current.run_id) == ("child", "child_run")
    target.message("hello", description="Test notice", source="subagent")
    assert (
        len(current.fork(namespace=default.namespace, run_id="parent_run").peek()) == 1
    )
    assert current.peek() == default.peek() == []
    if persistent:
        store.close()


def test_scoped_handle_keeps_explicit_store_precedence():
    library = ToolLibrary(name="parent", tools=[])
    bound = _inbox()
    handle = library.get_handle().for_tool(tool_name="host", agent_inbox=bound)
    with execution_context(agent_inbox=_inbox()):
        target = handle.get_agent_inbox(thread_id="thread", run_id="other")
    assert target.store is bound.store
    assert target.run_id == "other"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thread_id": "thread"},
        {"run_id": "run"},
        {"thread_id": "", "run_id": "run"},
        {"thread_id": 1, "run_id": "run"},
    ],
)
def test_scoped_handle_requires_complete_identity(kwargs):
    with pytest.raises(ValueError, match="thread_id and run_id"):
        ToolLibrary(name="parent", tools=[]).get_handle().get_agent_inbox(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_composition_delivers_to_parent_and_checkpoints(asynchronous):
    store = InMemoryCheckpointStore()
    parent = Agent(
        name="parent",
        model=_Model(
            _call("agent", {"name": "explorer", "message": "go"}, "delegate"), _text()
        ),
        tools=[AgentTool()],
        checkpoint_store=store,
    )
    child = Agent(
        name="explorer",
        model=_Model(_report(), _text()),
        tools=[SendParentMessageTool(parent.tool_library.get_handle())],
    )
    parent.tool_library.add(child)
    assert "send_parent_message" not in parent.tool_library.get_tool_names()
    assert "send_parent_message" in child.tool_library.get_tool_names()
    scope = ExecutionScope(namespace="parent", thread_id="thread", run_id="parent_run")
    result = (
        await parent.acall("delegate", scope=scope)
        if asynchronous
        else parent("delegate", scope=scope)
    )
    assert result == "done"
    state = store.load_state("parent", "thread", "parent_run")
    notices = [
        m
        for m in state["messages"]["items"]
        if m.get("metadata", {}).get("inbox_origin") == "subagent"
    ]
    assert len(notices) == 1
    assert "[explorer:parent_run:delegate:explorer] Found cause" in json.dumps(notices)
    assert "send_parent_message" in child.tool_library.get_tool_names()
    assert set(
        child.tool_library.get_tool_json_schemas()[0]["function"]["parameters"][
            "properties"
        ]
    ) == {"message"}


@pytest.mark.parametrize("persistent", [False, True])
def test_background_resume_uses_stable_task_id(persistent, tmp_path):
    inbox_store = (
        SQLiteAgentInboxStore(tmp_path / "inbox.sqlite3")
        if persistent
        else InMemoryAgentInboxStore()
    )
    inbox = _inbox(inbox_store)
    checkpoints = InMemoryCheckpointStore()
    library = ToolLibrary(
        name="parent", tools=[mf.tool_config(allow_background=True)(AgentTool())]
    )
    library.set_agent_inbox(inbox)
    child = Agent(
        name="explorer",
        model=_Model(_report("first"), _text(), _report("second", "send2"), _text()),
        tools=[SendParentMessageTool(library.get_handle())],
    )
    library.add(child)
    with execution_context(
        agent_inbox=inbox,
        checkpoint_store=checkpoints,
        thread_id="thread",
        namespace="parent",
        run_id="parent_run",
    ):
        result = library(
            [
                (
                    "delegate",
                    "agent",
                    {"name": "explorer", "message": "go", "run_in_background": True},
                )
            ]
        )
    assert result.tool_calls[0].error is None
    task_id = result.tool_calls[0].result.split("task_id='")[1].split("'")[0]
    result = library([("wait", "task_wait", {"task_id": task_id, "timeout": 5.0})])
    assert result.tool_calls[0].result == "done"
    with execution_context(agent_inbox=inbox, checkpoint_store=checkpoints):
        result = library(
            [("resume", "task_message", {"task_id": task_id, "message": "continue"})]
        )
        assert result.tool_calls[0].error is None
        result = library([("wait2", "task_wait", {"task_id": task_id, "timeout": 5.0})])
    assert result.tool_calls[0].result == "done"
    notices = [n for n in inbox.peek() if n.source == "incoming_message"]
    assert [n.metadata["content"] for n in notices] == [
        f"[explorer:{task_id}] first",
        f"[explorer:{task_id}] second",
    ]
    assert all(n.ref == task_id for n in notices)
    assert (
        library.get_handle().get_task_store().get(task_id).metadata["checkpoint_run_id"]
        != task_id
    )
    if persistent:
        reloaded = _inbox(SQLiteAgentInboxStore(tmp_path / "inbox.sqlite3"))
        assert len([n for n in reloaded.peek() if n.source == "incoming_message"]) == 2
        reloaded.store.close()
        inbox_store.close()


@pytest.mark.parametrize("background", [False, True])
def test_nested_child_targets_immediate_parent(background):
    checkpoints = InMemoryCheckpointStore()
    root_inbox = _inbox()
    root = ToolLibrary(
        name="parent", tools=[mf.tool_config(allow_background=True)(AgentTool())]
    )
    root.set_agent_inbox(root_inbox)
    middle = Agent(
        name="explorer",
        model=_Model(
            _call("agent", {"name": "builder", "message": "go"}, "leaf"),
            _report("middle"),
            _text(),
        ),
        tools=[AgentTool(), SendParentMessageTool(root.get_handle())],
    )
    leaf = Agent(
        name="builder",
        model=_Model(_report("leaf"), _text()),
        tools=[SendParentMessageTool(middle.tool_library.get_handle())],
    )
    middle.tool_library.add(leaf)
    root.add(middle)
    params = {"name": "explorer", "message": "go", "run_in_background": background}
    with execution_context(
        agent_inbox=root_inbox,
        checkpoint_store=checkpoints,
        namespace="parent",
        thread_id="thread",
        run_id="parent_run",
    ):
        result = root([("delegate", "agent", params)])
    assert result.tool_calls[0].error is None
    if background:
        task_id = result.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        result = root([("wait", "task_wait", {"task_id": task_id, "timeout": 5.0})])
    assert result.tool_calls[0].result == "done"
    incoming = [n for n in root_inbox.peek() if n.source == "incoming_message"]
    assert len(incoming) == 1 and incoming[0].metadata["content"].endswith("] middle")
    assert "] leaf" in str(middle.model.histories)
    assert "[builder:" in str(middle.model.histories)


@pytest.mark.asyncio
async def test_concurrent_compositions_keep_distinct_parent_routes():
    async def run(index):
        inbox = _inbox()
        library = ToolLibrary(name="parent", tools=[AgentTool()])
        library.set_agent_inbox(inbox)
        child = Agent(
            name="explorer",
            model=_Model(_report(str(index)), _text()),
            tools=[SendParentMessageTool(library.get_handle())],
        )
        library.add(child)
        with execution_context(
            agent_inbox=inbox,
            thread_id="thread",
            namespace="parent",
            run_id=f"parent_{index}",
        ):
            result = await library.acall(
                [("delegate", "agent", {"name": "explorer", "message": "go"})]
            )
        assert result.tool_calls[0].error is None
        target = inbox.fork(run_id=f"parent_{index}")
        assert [n.metadata["content"] for n in target.peek()] == [
            f"[explorer:parent_{index}:delegate:explorer] {index}"
        ]

    await asyncio.gather(*(run(i) for i in range(8)))


@pytest.mark.asyncio
async def test_coding_factory_composes_nested_messages_with_managed_storage(
    tmp_path, monkeypatch
):
    import msgflux.coding.factory as factory_module
    from msgflux.coding.config import CodingAgentCatalog, CodingAgentOptions
    from msgflux.runtime.service import ServiceThread

    responses = {
        "coding": [
            _call("agent", {"name": "explorer", "message": "investigate"}, "explore"),
            _text(),
        ],
        "explorer": [
            _call("agent", {"name": "builder", "message": "inspect"}, "build"),
            _report("explorer result"),
            _text(),
        ],
        "builder": [_report("builder result"), _text()],
    }
    models = {}

    def create_model(options):
        models[options.agent_id] = _Model(*responses[options.agent_id])
        return models[options.agent_id]

    monkeypatch.setattr(factory_module, "_create_model", create_model)
    options = {
        name: CodingAgentOptions(
            agent_id=name,
            model="openai/example",
            reasoning_effort="medium",
            tools_active=(),
            subagents=children,
        )
        for name, children in [
            ("coding", ("explorer",)),
            ("explorer", ("builder",)),
            ("builder", ()),
        ]
    }
    session = await factory_module.create_coding_session_factory(
        options=options["coding"],
        catalog=CodingAgentCatalog(options),
        config_path=tmp_path / "config.toml",
        runtime_dir=tmp_path / "runtime",
        agent_dir=tmp_path / "agent-state",
    )(ServiceThread("factory-thread", "coding", str(tmp_path)))
    main = session.agent
    try:
        assert "send_parent_message" not in main.tool_library.get_tool_names()
        child = main.tool_library.get_tool_definition("explorer").executor.impl
        leaf = child.tool_library.get_tool_definition("builder").executor.impl
        assert "send_parent_message" in child.tool_library.get_tool_names()
        assert "send_parent_message" in leaf.tool_library.get_tool_names()
        scope = ExecutionScope(
            namespace="coding", thread_id="factory-thread", run_id="factory-run"
        )
        assert await main.acall("delegate", scope=scope) == "done"
        # The real factory's managed backend must contain only the immediate child's notice.
        store = main._bind_resources(scope.thread_id).checkpoint_store
        state = store.load_state("coding", scope.thread_id, scope.run_id)
        incoming = [
            item
            for item in state["messages"]["items"]
            if item.get("metadata", {}).get("inbox_origin") == "subagent"
        ]
        assert len(incoming) == 1
        assert "explorer result" in str(incoming)
        assert "builder result" not in str(incoming)
        assert "builder result" in str(models["explorer"].histories)
    finally:
        await session.on_close()
