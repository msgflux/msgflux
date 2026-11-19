"""Resolve editable tool text through the owning agent's local registry."""

from msgflux.nn import Agent


def get_tool_text_parameter(agent: Agent, name: str, field: str):
    if agent.tool_library is None:
        raise ValueError(f"No directly registered local tool named {name!r}")
    return agent.tool_library.text_parameter(name, field)


def get_tool_description(agent: Agent, name: str) -> str:
    return get_tool_text_parameter(agent, name, "description").data


def set_tool_description(agent: Agent, name: str, text: str) -> None:
    set_tool_text(agent, name, "description", text)


def set_tool_text(agent: Agent, name: str, field: str, text: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"A non-empty tool {field} is required")
    get_tool_text_parameter(agent, name, field).copy_to_data(text)
