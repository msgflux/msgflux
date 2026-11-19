"""Tool description targets rewrite one Agent's local tool catalog entry."""

from unittest.mock import Mock

import msgspec
import pytest

from msgflux.nn import Agent
from msgflux.nn.modules.tool.implementations import MCPTool as MCPToolExecutor
from msgflux.optim.tool_target import get_tool_description, set_tool_description
from msgflux.protocols.mcp.types import MCPTool as MCPToolInfo
from msgflux.tools.builtin import AgentTool
from msgflux.tools.catalog import NativeToolBinding

DECLARED = "Look up a value."


def _model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


def _agent(*tools):
    return Agent(name="main", model=_model(), tools=list(tools))


def _lookup_tool(**config):
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    if config:
        lookup.tool_config = dict(config)
    return lookup


def _descriptions(view):
    return {entry.name: entry.description for entry in view.entries}


def test_update_reaches_catalog_and_executable_definition_projection():
    tool = _lookup_tool()
    agent = _agent(tool)
    library = agent.tool_library
    before = library.get_tool_definition("lookup")

    set_tool_description(agent, "lookup", "Resolve a key, case-sensitively.")

    after = library.get_tool_definition("lookup")
    # Executable definition: only the description is replaced.
    assert after.name == before.name == "lookup"
    assert after.executor is before.executor
    assert after.input_schema == before.input_schema
    assert after.annotations == before.annotations
    assert after.kind == before.kind
    assert after.dispatch == before.dispatch
    assert after.metadata == before.metadata
    assert after.native_bindings == before.native_bindings

    # Canonical catalog: thread-scoped view, portable schema, and legacy catalog.
    view = library.get_tool_catalog_view(thread_id="thread-1")
    assert _descriptions(view)["lookup"] == "Resolve a key, case-sensitively."
    schemas = {
        schema["function"]["name"]: schema["function"]
        for schema in view.portable_schemas()
    }
    assert schemas["lookup"]["description"] == "Resolve a key, case-sensitively."
    assert schemas["lookup"]["parameters"] == dict(before.input_schema)
    portable = {
        schema["function"]["name"]: schema["function"]
        for schema in library.get_tool_json_schemas()
    }
    assert portable["lookup"]["description"] == "Resolve a key, case-sensitively."
    legacy = {spec.name: spec for spec in library.get_tool_catalog().tools}
    assert legacy["lookup"].description == "Resolve a key, case-sensitively."
    assert legacy["lookup"].parameters == dict(before.input_schema)

    # Execution is preserved by the replacement.
    assert library.library["lookup"](query="answer") == "answer"
    assert after.executor(query="answer") == "answer"


def test_remote_grouped_and_deferred_tools_are_not_targetable():
    def plain(query: str) -> str:
        """Plain lookup."""
        return query

    deferred = _lookup_tool(defer_loading=True)
    agent = _agent(deferred, plain)
    assert get_tool_description(agent, "plain") == "Plain lookup."
    with pytest.raises(ValueError, match="No directly registered local tool"):
        get_tool_description(agent, "lookup")
    with pytest.raises(ValueError, match="portable local declaration"):
        get_tool_description(agent, "tool_search")

    worker = Agent(name="worker", model=_model(), tools=[plain])
    grouped = _agent(AgentTool(), worker, plain)
    with pytest.raises(ValueError, match="portable local declaration"):
        get_tool_description(grouped, "agent")
    with pytest.raises(ValueError, match="No directly registered local tool"):
        get_tool_description(grouped, "worker")

    client = Mock()
    remote_info = MCPToolInfo(name="read_file", description="Read.", inputSchema={})
    remote = MCPToolExecutor(
        name="read_file",
        mcp_client=client,
        mcp_tool_info=remote_info,
        namespace="fs",
    )
    agent.tool_library.add(remote)
    with pytest.raises(ValueError, match="portable local declaration"):
        get_tool_description(agent, "fs__read_file")


def test_native_binding_tools_are_not_targetable():
    def shell(query: str) -> str:
        """Run a command."""
        return query

    shell.native_bindings = (
        NativeToolBinding(
            provider="openai",
            api_mode="responses",
            kind="shell",
            execution="provider",
        ),
    )
    agent = _agent(shell)

    with pytest.raises(ValueError, match="portable local declaration"):
        get_tool_description(agent, "shell")
    with pytest.raises(ValueError, match="portable local declaration"):
        set_tool_description(agent, "shell", "Rewrite.")


def test_unknown_tools_and_missing_descriptions_are_explicit():
    agent = _agent(_lookup_tool())
    definition = agent.tool_library.get_tool_definition("lookup")
    agent.tool_library.registry.replace(
        msgspec.structs.replace(definition, description=None)
    )

    with pytest.raises(ValueError, match="No directly registered local tool"):
        get_tool_description(agent, "missing")
    assert get_tool_description(agent, "lookup") == ""
    with pytest.raises(ValueError, match="No directly registered local tool"):
        set_tool_description(agent, "missing", "Rewrite.")


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 3])
def test_non_text_and_empty_descriptions_are_rejected(text):
    agent = _agent(_lookup_tool())

    with pytest.raises(ValueError, match="non-empty tool description"):
        set_tool_description(agent, "lookup", text)

    assert get_tool_description(agent, "lookup") == DECLARED


def test_documented_factory_opts_in_only_the_requested_tool_fields():
    from examples.optimize_tool_description import build_agent

    plain = build_agent(_model()).tool_library
    assert plain.description_parameter("lookup").requires_grad
    assert not plain.usage_guidance_parameter("lookup").requires_grad
    guided = build_agent(_model(), train_usage_guidance=True)
    parameter = guided.tool_library.usage_guidance_parameter("lookup")
    assert parameter.requires_grad
    assert parameter.data in guided.get_system_prompt()
