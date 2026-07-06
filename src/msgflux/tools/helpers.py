from importlib import import_module
from typing import Any, Callable, Mapping

RUNTIME_BACKGROUND_PARAM = "run_in_background"


def should_copy_injected_messages(tool: Callable, config: Mapping[str, Any]) -> bool:
    if not config.get("inject_messages", False):
        return False

    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(getattr(tool, "impl", tool), agent_type)


def uses_handle_injection(config: Mapping[str, Any]) -> bool:
    return config.get("inject_handle", False)


def is_agent_tool_impl(impl: Any) -> bool:
    if getattr(impl, "is_agent_tool", False):
        return True
    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(impl, agent_type)
