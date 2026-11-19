"""Read and update the portable description owned by one Agent's tool registry."""

import msgspec

from msgflux.nn import Agent, MCPTool
from msgflux.tools.types import ToolBucket


def _local_definition(agent: Agent, name: str):
    library = agent.tool_library
    if library is None or name not in library.library:
        raise ValueError(f"No directly registered local tool named {name!r}")
    definition = library.get_tool_definition(name)
    if (
        isinstance(definition.executor, MCPTool)
        or isinstance(getattr(definition.executor, "impl", None), ToolBucket)
        or definition.native_bindings
    ):
        raise ValueError(f"Tool {name!r} requires a portable local declaration")
    return definition


def get_tool_description(agent: Agent, name: str) -> str:
    return _local_definition(agent, name).description or ""


def set_tool_description(agent: Agent, name: str, text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A non-empty tool description is required")
    definition = _local_definition(agent, name)
    agent.tool_library.registry.replace(
        msgspec.structs.replace(definition, description=text)
    )
