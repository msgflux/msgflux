from unittest.mock import Mock

import msgflux as mf
import pytest
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.context import execution_context
from msgflux.tools.builtin import AgentTool


class _ScriptedModel:
    def __init__(self, *texts: str):
        self.model_type = "chat_completion"
        self._texts = list(texts or ("ok",))
        self.calls = []

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
    tool = AgentTool([reviewer])
    library = ToolLibrary(name="lib", tools=[tool])

    schema = library.get_tool_json_schemas()[0]

    assert schema["function"]["name"] == "agent"
    properties = schema["function"]["parameters"]["properties"]
    assert set(properties) == {"name", "message"}


def test_agent_tool_collects_agent_usage_guidance():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    reviewer.tool_config = {"usage_guidance": "Use for code review."}
    planner.usage_guidance = "Use for planning."

    tool = AgentTool([reviewer, planner])

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


def test_agent_tool_rejects_background_agent_on_initialization():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    reviewer.tool_config = {"background": True}

    with pytest.raises(ValueError, match="Bucket-captured tools cannot use"):
        AgentTool([reviewer])


def test_agent_tool_rejects_on_demand_agent_on_initialization():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    reviewer.tool_config = {"on_demand": True}

    with pytest.raises(ValueError, match="does not match this bucket's capture rule"):
        AgentTool([reviewer])


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

    assert "Bucket-captured tools cannot use `background=True`" in error
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

    assert "Bucket-captured tools cannot use `background=True`" in error
    assert "reviewer" in library.library
    assert "agent" not in library.library


def test_agent_tool_dispatches_to_selected_agent():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    library = ToolLibrary(name="lib", tools=[AgentTool([reviewer, planner])])

    response = library([("call_1", "agent", {"name": "reviewer", "message": "Go"})])

    assert response.tool_calls[0].result == "reviewed"
    assert len(reviewer.model.calls) == 1
    assert planner.model.calls == []


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
    assert recorder.calls[0]["messages"] == messages
    assert recorder.calls[0]["vars"] == vars


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
    library = ToolLibrary(name="lib", tools=[AgentTool([reviewer])])

    response = library([("call_1", "agent", {"name": "missing", "message": "Go"})])

    assert response.tool_calls[0].result is None
    assert "Agent `missing` not found" in response.tool_calls[0].error
    assert "reviewer" in response.tool_calls[0].error


def test_agent_tool_captures_on_demand_agents_after_tool_search():
    reviewer = mf.tool_config(on_demand=True)(
        Agent(name="reviewer", model=_mock_model("reviewed"))
    )
    reviewer.tool_config.usage_guidance = "Use for code review."
    library = ToolLibrary(name="lib", tools=[AgentTool(), reviewer])

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
            ]
        )
        .tool_calls[0]
        .result
    )
    schema_names_after = [
        schema["function"]["name"] for schema in library.get_tool_json_schemas()
    ]
    response = library([("call_3", "agent", {"name": "reviewer", "message": "Go"})])

    assert before_search.tool_calls[0].result is None
    assert "Agent `reviewer` not found" in before_search.tool_calls[0].error
    assert schema_names_before == ["agent", "tool_search"]
    assert search["loaded"] == ["reviewer"]
    assert search["descriptions"][0]["name"] == "reviewer"
    assert schema_names_after == ["agent"]
    assert response.tool_calls[0].result == "reviewed"
    assert "reviewer: Use for code review." in library.library["agent"].usage_guidance


def test_agent_tool_background_run_uses_task_id_as_child_run_id():
    store = InMemoryCheckpointStore()
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    agent_tool = mf.tool_config(allow_background=True)(AgentTool([reviewer]))
    library = ToolLibrary(name="lib", tools=[agent_tool])

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
    agent_tool = mf.tool_config(allow_background=True)(AgentTool([reviewer]))
    library = ToolLibrary(name="lib", tools=[agent_tool])

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
