import asyncio
from dataclasses import replace
from unittest.mock import Mock

import msgflux as mf
import pytest
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.context import execution_context
from msgflux.runtime.events import EventType
from msgflux.tools import BUILTIN_TOOL_USAGE_GUIDANCE, apply_tool_guidance
from msgflux.tools.builtin import AgentTool


class _ScriptedModel(BaseModel):
    def __init__(self, *texts: str, model_id: str = "scripted"):
        self.model_type = "chat_completion"
        self.model_id = model_id
        self.provider = "mock"
        self._texts = list(texts or ("ok",))
        self.calls = []

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._texts:
            raise AssertionError("Scripted model exhausted.")
        return _response(self._texts.pop(0))

    async def acall(self, **kwargs):
        return self(**kwargs)


def _response(text: str) -> Mock:
    resp = Mock(spec=ModelResponse)
    resp.response_type = "text_generation"
    resp.consume.return_value = text
    resp.data = text
    resp.reasoning = None
    resp.metadata = {}
    return resp


def _mock_model(*texts: str) -> _ScriptedModel:
    return _ScriptedModel(*texts)


def _model_gateway(*, fallback: bool = True) -> ModelGateway:
    return ModelGateway(
        models=[
            {
                "model_name": "fast",
                "model": _ScriptedModel("fast response", model_id="fast-model"),
                "description": "Fast for clear, repeatable tasks.",
            },
            {
                "model_name": "deep",
                "model": _ScriptedModel("deep response", model_id="deep-model"),
                "description": "Deep reasoning for complex tasks.",
            },
        ],
        fallback=fallback,
    )


def _extract_task_id(result: str) -> str:
    return result.split("task_id='", maxsplit=1)[1].split("'", maxsplit=1)[0]


class _RecordingAgent:
    tool_kind = "agent"
    description = "Record calls."

    def __init__(self, name: str = "recorder"):
        self.name = name
        self.calls = []

    def __call__(self, message, **kwargs):
        self.calls.append({"message": message, **kwargs})
        return "recorded"

    async def acall(self, message, **kwargs):
        return self(message, **kwargs)

    def get_module_name(self):
        return self.name

    def get_module_description(self):
        return self.description


def test_agent_tool_exposes_single_agent_tool_with_name_and_message_params():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    schema = library.get_tool_json_schemas()[0]

    assert schema["function"]["name"] == "agent"
    properties = schema["function"]["parameters"]["properties"]
    assert set(properties) == {"name", "message"}
    assert schema["function"]["parameters"]["required"] == ["name", "message"]


def test_agent_tool_description_lists_available_agents():
    reviewer = Agent(
        name="reviewer",
        model=_mock_model("reviewed"),
        description="Reviews drafts for clarity.",
    )

    tool = AgentTool()
    ToolLibrary(name="lib", tools=[tool, reviewer])

    assert (
        tool.description == "Available agents:\n- reviewer: Reviews drafts for clarity."
    )


def test_agent_tool_exposes_gateway_models_with_descriptions():
    gateway = _model_gateway()
    reviewer = Agent(
        name="reviewer",
        model=gateway,
        description="Reviews code and architecture.",
    )
    tool = AgentTool()
    library = ToolLibrary(name="lib", tools=[tool, reviewer])

    schema = library.get_tool_json_schemas()[0]
    properties = schema["function"]["parameters"]["properties"]

    assert set(properties) == {"name", "message", "model"}
    assert schema["function"]["parameters"]["required"] == [
        "name",
        "message",
        "model",
    ]
    assert {option["type"] for option in properties["model"]["anyOf"]} == {
        "string",
        "null",
    }
    assert "- reviewer: Reviews code and architecture." in tool.description
    assert "  - fast: Fast for clear, repeatable tasks." in tool.description
    assert "  - deep: Deep reasoning for complex tasks." in tool.description


def test_agent_tool_requires_descriptions_for_gateway_models():
    gateway = ModelGateway(
        models=[
            {
                "model_name": "undescribed",
                "model": _ScriptedModel(model_id="undescribed"),
            }
        ]
    )
    reviewer = Agent(name="reviewer", model=gateway)
    library = ToolLibrary(name="lib", tools=[AgentTool()])

    with pytest.raises(ValueError, match="without a description"):
        library.add(reviewer)

    assert library.get_tool_names() == ["agent"]
    assert library.get_tool_json_schemas()[0]["function"]["parameters"][
        "properties"
    ].keys() == {"name", "message"}


def test_agent_tool_removes_model_selector_after_last_gateway_agent_is_removed():
    reviewer = Agent(name="reviewer", model=_model_gateway())
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    assert (
        "model"
        in library.get_tool_json_schemas()[0]["function"]["parameters"]["properties"]
    )

    library.remove("reviewer")

    assert (
        "model"
        not in library.get_tool_json_schemas()[0]["function"]["parameters"][
            "properties"
        ]
    )


def test_agent_tool_builtin_usage_guidance_is_opt_in_and_survives_capture():
    assert (
        ToolLibrary(name="default", tools=[AgentTool()]).get_tool_usage_guidance() == []
    )

    [agent_tool] = apply_tool_guidance([AgentTool()])
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    library = ToolLibrary(name="lib", tools=[agent_tool])

    expected_guidance = [
        {
            "name": "agent",
            "display_name": "Agent",
            "guidance": BUILTIN_TOOL_USAGE_GUIDANCE["agent"],
        }
    ]
    assert library.get_tool_usage_guidance() == expected_guidance

    library.add(reviewer)

    assert library.get_tool_usage_guidance() == expected_guidance


def test_agent_tool_collects_agent_usage_guidance():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    reviewer.tool_config = {"usage_guidance": "Use for code review."}
    planner.usage_guidance = "Use for planning."

    tool = AgentTool()
    ToolLibrary(name="lib", tools=[tool, reviewer, planner])

    assert "reviewer: Use for code review." in tool.usage_guidance
    assert "planner: Use for planning." in tool.usage_guidance


def test_agent_tool_can_start_empty_and_capture_agents_from_library():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    reviewer.tool_config = {"usage_guidance": "Use for code review."}
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    schema_names = [
        schema["function"]["name"] for schema in library.get_tool_json_schemas()
    ]
    agent_schema = next(
        schema
        for schema in library.get_tool_json_schemas()
        if schema["function"]["name"] == "agent"
    )

    assert schema_names == ["agent"]
    assert "reviewer" not in library.library
    assert "reviewer: Use for code review." in library.library["agent"].usage_guidance
    assert "reviewer" in agent_schema["function"]["description"]

    response = library([("call_1", "agent", {"name": "reviewer", "message": "Go"})])

    assert response.tool_calls[0].result == "reviewed"


def test_agent_tool_rejects_agents_in_constructor():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))

    with pytest.raises(TypeError, match="unexpected keyword argument 'agents'"):
        AgentTool(agents=[reviewer])


def test_agent_tool_captures_registered_agents_when_added_later():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    library = ToolLibrary(name="lib", tools=[reviewer])
    library.add(AgentTool())

    schema_names = [
        schema["function"]["name"] for schema in library.get_tool_json_schemas()
    ]
    response = library([("call_1", "agent", {"name": "reviewer", "message": "Go"})])

    assert schema_names == ["agent"]
    assert "reviewer" not in library.library
    assert response.tool_calls[0].result == "reviewed"


def test_agent_tool_rejects_background_agent_capture():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    reviewer.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[AgentTool()])

    try:
        library.add(reviewer)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("AgentTool should reject background captured agents.")

    assert "Bucket-captured tools cannot define model-loop behavior" in error
    assert "reviewer" not in library.library


def test_agent_tool_rejects_existing_allow_background_agent_capture():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    reviewer.tool_config = {"allow_background": True}
    library = ToolLibrary(name="lib", tools=[reviewer])

    try:
        library.add(AgentTool())
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("AgentTool should reject allow-background agents.")

    assert "Bucket-captured tools cannot define model-loop behavior" in error
    assert "reviewer" in library.library
    assert "agent" not in library.library


def test_agent_tool_dispatches_to_selected_agent():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer, planner])

    response = library([("call_1", "agent", {"name": "reviewer", "message": "Go"})])

    assert response.tool_calls[0].result == "reviewed"
    assert len(reviewer.model.calls) == 1
    assert planner.model.calls == []


def test_agent_tool_dispatches_to_selected_gateway_model():
    gateway = _model_gateway()
    reviewer = Agent(name="reviewer", model=gateway)
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    response = library(
        [
            (
                "call_1",
                "agent",
                {"name": "reviewer", "message": "Review this", "model": "deep"},
            )
        ]
    )

    assert response.tool_calls[0].result == "deep response"
    assert gateway.models[0].calls == []
    assert len(gateway.models[1].calls) == 1
    assert response.tool_calls[0].parameters == {
        "name": "reviewer",
        "message": "Review this",
        "model": "deep",
    }


@pytest.mark.asyncio
async def test_agent_tool_dispatches_concurrent_gateway_model_selections():
    gateway = _model_gateway()
    reviewer = Agent(name="reviewer", model=gateway)
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    fast, deep = await asyncio.gather(
        library.acall(
            [
                (
                    "call_fast",
                    "agent",
                    {"name": "reviewer", "message": "Classify", "model": "fast"},
                )
            ]
        ),
        library.acall(
            [
                (
                    "call_deep",
                    "agent",
                    {"name": "reviewer", "message": "Analyze", "model": "deep"},
                )
            ]
        ),
    )

    assert fast.tool_calls[0].result == "fast response"
    assert deep.tool_calls[0].result == "deep response"
    assert len(gateway.models[0].calls) == 1
    assert len(gateway.models[1].calls) == 1


def test_agent_tool_ignores_model_for_agent_without_gateway():
    gateway_reviewer = Agent(name="gateway_reviewer", model=_model_gateway())
    recorder = _RecordingAgent()
    library = ToolLibrary(
        name="lib",
        tools=[AgentTool(), gateway_reviewer, recorder],
    )

    response = library(
        [
            (
                "call_1",
                "agent",
                {"name": "recorder", "message": "Review this", "model": "deep"},
            )
        ]
    )

    assert response.tool_calls[0].result == "recorded"
    assert recorder.calls[0]["message"] == "Review this"
    assert "model_preference" not in recorder.calls[0]


def test_agent_tool_rejects_unknown_gateway_model():
    reviewer = Agent(name="reviewer", model=_model_gateway())
    tool = AgentTool()
    library = ToolLibrary(name="lib", tools=[tool, reviewer])

    with pytest.raises(ValueError, match="Unknown model `missing`"):
        tool(
            "reviewer",
            "Review this",
            model="missing",
            handle=library.get_handle().for_tool(tool_name="agent"),
        )


def test_agent_tool_before_tool_can_rewrite_model_message():
    recorder = _RecordingAgent()
    owner = Agent(name="owner", model=_mock_model())
    Hook(
        event="before_tool",
        handler=lambda event: replace(
            event,
            arguments={**event.arguments, "message": "Rewritten review task"},
        ),
    ).register(owner)
    library = ToolLibrary(name="lib", tools=[AgentTool(), recorder])
    library.set_lifecycle_owner(owner)

    response = library(
        [("call_1", "agent", {"name": "recorder", "message": "Original task"})]
    )

    assert response.tool_calls[0].parameters == {
        "name": "recorder",
        "message": "Rewritten review task",
    }
    assert recorder.calls[0]["message"] == "Rewritten review task"


@pytest.mark.asyncio
async def test_agent_tool_events_separate_visible_and_runtime_arguments():
    recorder = _RecordingAgent()
    recorder.tool_config = {"inject_messages": True, "inject_vars": True}
    library = ToolLibrary(name="lib", tools=[AgentTool(), recorder])

    events = [
        event
        async for event in library.stream_events(
            tool_callings=[
                (
                    "call_1",
                    "agent",
                    {"name": "recorder", "message": "Review this"},
                )
            ],
            messages=[{"role": "user", "content": "Parent context"}],
            vars={"tenant": "acme"},
        )
    ]
    starts = [event for event in events if event.type == EventType.TOOL_START]

    assert starts[0].data["arguments"] == {
        "name": "recorder",
        "message": "Review this",
    }
    assert starts[1].data["arguments"] == {"message": "Review this"}
    assert all(
        not (
            {"handle", "messages", "vars", "scope", "tool_call_id"}
            & set(event.data["arguments"])
        )
        for event in starts
    )


def test_agent_tool_injects_messages_and_vars_without_exposing_them():
    recorder = _RecordingAgent()
    recorder.tool_config = {"inject_messages": True, "inject_vars": True}
    library = ToolLibrary(name="lib", tools=[AgentTool(), recorder])
    messages = [{"role": "user", "content": "history"}]
    vars = {"tenant": "acme"}

    schema = library.get_tool_json_schemas()[0]
    response = library(
        [("call_1", "agent", {"name": "recorder", "message": "Go"})],
        messages=messages,
        vars=vars,
    )

    assert set(schema["function"]["parameters"]["properties"]) == {"name", "message"}
    assert response.tool_calls[0].result == "recorded"
    assert response.tool_calls[0].parameters == {
        "name": "recorder",
        "message": "Go",
    }
    assert recorder.calls[0]["messages"] == messages
    assert recorder.calls[0]["vars"] == vars


def test_agent_tool_child_agent_inherits_parent_messages_when_configured():
    reviewer = mf.tool_config(runtime_inputs=["messages"])(
        Agent(name="reviewer", model=_mock_model("reviewed"))
    )
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])
    messages = [{"role": "user", "content": "parent history"}]

    response = library(
        [("call_1", "agent", {"name": "reviewer", "message": "Review this"})],
        messages=messages,
    )

    assert response.tool_calls[0].result == "reviewed"
    inherited = reviewer.model.calls[0]["messages"]
    assert inherited is not messages
    assert inherited.to_chatml()[0] == messages[0]


def test_agent_tool_uses_canonical_agent_name_for_default_scope():
    reviewer = mf.tool_config(name_override="review_alias")(_RecordingAgent("reviewer"))
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    response = library([("call_1", "agent", {"name": "review_alias", "message": "Go"})])

    assert response.tool_calls[0].result == "recorded"
    assert reviewer.calls[0]["scope"].namespace == "reviewer"


def test_agent_tool_injects_runtime_kwargs_only_for_selected_agent_config():
    contextual = _RecordingAgent("contextual")
    contextual.tool_config = {"inject_messages": True, "inject_vars": True}
    plain = _RecordingAgent("plain")
    library = ToolLibrary(name="lib", tools=[AgentTool(), contextual, plain])
    messages = [{"role": "user", "content": "history"}]
    vars = {"tenant": "acme"}

    plain_response = library(
        [("call_1", "agent", {"name": "plain", "message": "Go"})],
        messages=messages,
        vars=vars,
    )
    contextual_response = library(
        [("call_2", "agent", {"name": "contextual", "message": "Go"})],
        messages=messages,
        vars=vars,
    )

    assert plain_response.tool_calls[0].result == "recorded"
    assert contextual_response.tool_calls[0].result == "recorded"
    assert "messages" not in plain.calls[0]
    assert "vars" not in plain.calls[0]
    assert plain.calls[0]["scope"].namespace == "plain"
    assert contextual.calls[0]["messages"] == messages
    assert contextual.calls[0]["messages"] is not messages
    assert contextual.calls[0]["vars"] == vars
    assert contextual.calls[0]["scope"].namespace == "contextual"


def test_agent_tool_filters_injected_vars_list_for_selected_agent():
    recorder = _RecordingAgent()
    recorder.tool_config = {"inject_vars": ["tenant"]}
    library = ToolLibrary(name="lib", tools=[AgentTool(), recorder])

    response = library(
        [("call_1", "agent", {"name": "recorder", "message": "Go"})],
        vars={"tenant": "acme", "secret": "hidden"},
    )

    assert response.tool_calls[0].result == "recorded"
    assert recorder.calls[0]["vars"] == {"tenant": "acme"}


def test_agent_tool_injected_vars_list_requires_selected_agent_vars():
    recorder = _RecordingAgent()
    recorder.tool_config = {"inject_vars": ["tenant"]}
    library = ToolLibrary(name="lib", tools=[AgentTool(), recorder])

    response = library(
        [("call_1", "agent", {"name": "recorder", "message": "Go"})],
        vars={},
    )

    assert response.tool_calls[0].result is None
    assert "The agent `recorder` requires the injected parameter `tenant`" in (
        response.tool_calls[0].error
    )


def test_agent_tool_rejects_unknown_agent_name():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

    response = library([("call_1", "agent", {"name": "missing", "message": "Go"})])

    assert response.tool_calls[0].result is None
    assert "Agent `missing` not found" in response.tool_calls[0].error
    assert "reviewer" in response.tool_calls[0].error


def test_deferred_agent_load_does_not_mutate_agent_tool():
    reviewer = mf.tool_config(defer_loading=True)(
        Agent(name="reviewer", model=_mock_model("reviewed"))
    )
    reviewer.tool_config.usage_guidance = "Use for code review."
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])
    messages = ChatMessages(thread_id="review-thread")
    agent_tool = library.library["agent"]
    description_before = agent_tool.description
    schema_before = agent_tool.get_json_schema()

    before_search = library(
        [("call_1", "agent", {"name": "reviewer", "message": "Go"})]
    )
    schema_names_before = [
        schema["function"]["name"] for schema in library.get_tool_json_schemas()
    ]

    search = (
        library(
            [
                (
                    "call_2",
                    "tool_search",
                    {"select": ["reviewer"], "description": True},
                )
            ],
            messages=messages,
        )
        .tool_calls[0]
        .result
    )
    visible_after = [
        tool.name for tool in library.get_tool_catalog(messages).portable_tools()
    ]
    response = library(
        [("call_3", "reviewer", {"message": "Go"})],
        messages=messages,
    )

    assert before_search.tool_calls[0].result is None
    assert "Agent `reviewer` not found" in before_search.tool_calls[0].error
    assert schema_names_before == ["agent", "tool_search"]
    assert search["loaded"] == ["reviewer"]
    assert search["descriptions"][0]["name"] == "reviewer"
    assert visible_after == ["agent", "reviewer"]
    assert response.tool_calls[0].result == "reviewed"
    assert "reviewer" not in library.library
    assert agent_tool.description == description_before
    assert agent_tool.get_json_schema() == schema_before


def test_agent_tool_background_run_uses_task_id_as_child_run_id():
    store = InMemoryCheckpointStore()
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    agent_tool = mf.tool_config(allow_background=True)(AgentTool())
    library = ToolLibrary(name="lib", tools=[agent_tool, reviewer])

    with execution_context(
        thread_id="user_42",
        namespace="root",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library(
            [
                (
                    "call_1",
                    "agent",
                    {
                        "name": "reviewer",
                        "message": "Review this",
                        "run_in_background": True,
                    },
                )
            ]
        )

    task_id = _extract_task_id(dispatch.tool_calls[0].result)
    wait = library([("call_2", "task_wait", {"task_id": task_id, "timeout": 1.0})])

    assert "`task_activity`" in dispatch.tool_calls[0].result
    assert "`task_message`" in dispatch.tool_calls[0].result
    assert wait.tool_calls[0].result == "reviewed"
    assert store.load_state("reviewer", "user_42", task_id)["status"] == "completed"


def test_agent_tool_background_task_message_resumes_selected_agent():
    store = InMemoryCheckpointStore()
    model = _mock_model("first", "second")
    reviewer = Agent(name="reviewer", model=model)
    agent_tool = mf.tool_config(allow_background=True)(AgentTool())
    library = ToolLibrary(name="lib", tools=[agent_tool, reviewer])

    with execution_context(
        thread_id="user_42",
        namespace="root",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library(
            [
                (
                    "call_1",
                    "agent",
                    {
                        "name": "reviewer",
                        "message": "First",
                        "run_in_background": True,
                    },
                )
            ]
        )

    task_id = _extract_task_id(dispatch.tool_calls[0].result)
    wait = library([("call_2", "task_wait", {"task_id": task_id, "timeout": 1.0})])
    assert wait.tool_calls[0].result == "first"

    with execution_context(checkpoint_store=store):
        resumed = library(
            [("call_3", "task_message", {"task_id": task_id, "message": "Second"})]
        )
        assert resumed.tool_calls[0].result["status"] == "resumed"
        wait_resumed = library(
            [("call_4", "task_wait", {"task_id": task_id, "timeout": 1.0})]
        )

    assert wait_resumed.tool_calls[0].result == "second"
    old_state = store.load_state("reviewer", "user_42", task_id)
    assert old_state["status"] == "completed"
    task_state = library([("call_5", "task_status", {"task_id": task_id})])
    resumed_run_id = task_state.tool_calls[0].result["metadata"]["checkpoint_run_id"]
    assert resumed_run_id != task_id
    new_state = store.load_state("reviewer", "user_42", resumed_run_id)
    assert new_state["status"] == "completed"


def test_agent_tool_background_resume_preserves_gateway_model_selection():
    store = InMemoryCheckpointStore()
    fast_model = _ScriptedModel("unexpected", model_id="fast-model")
    deep_model = _ScriptedModel("first", "second", model_id="deep-model")
    gateway = ModelGateway(
        models=[
            {
                "model_name": "fast",
                "model": fast_model,
                "description": "Fast for clear, repeatable tasks.",
            },
            {
                "model_name": "deep",
                "model": deep_model,
                "description": "Deep reasoning for complex tasks.",
            },
        ]
    )
    reviewer = Agent(name="reviewer", model=gateway)
    agent_tool = mf.tool_config(allow_background=True)(AgentTool())
    library = ToolLibrary(name="lib", tools=[agent_tool, reviewer])

    with execution_context(
        thread_id="user_gateway",
        namespace="root",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library(
            [
                (
                    "call_1",
                    "agent",
                    {
                        "name": "reviewer",
                        "message": "First",
                        "model": "deep",
                        "run_in_background": True,
                    },
                )
            ]
        )

    task_id = _extract_task_id(dispatch.tool_calls[0].result)
    first_wait = library(
        [("call_2", "task_wait", {"task_id": task_id, "timeout": 1.0})]
    )
    assert first_wait.tool_calls[0].result == "first"

    with execution_context(checkpoint_store=store):
        resumed = library(
            [("call_3", "task_message", {"task_id": task_id, "message": "Second"})]
        )
        assert resumed.tool_calls[0].result["status"] == "resumed"
        second_wait = library(
            [("call_4", "task_wait", {"task_id": task_id, "timeout": 1.0})]
        )

    assert second_wait.tool_calls[0].result == "second"
    assert fast_model.calls == []
    assert len(deep_model.calls) == 2
