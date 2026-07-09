from importlib import import_module
from typing import Any, Callable, Mapping

BACKGROUND_TASK_TOOL_KIND = "background"
TOOL_BUCKET_KIND = "bucket"
RUNTIME_BACKGROUND_PARAM = "run_in_background"
RESERVED_TOOL_KINDS = {BACKGROUND_TASK_TOOL_KIND}


def is_reserved_tool_kind(config: Mapping[str, Any]) -> bool:
    return config.get("tool_kind") in RESERVED_TOOL_KINDS


def is_background_capable(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("background", False) or config.get("allow_background", False)
    )


def should_copy_injected_messages(tool: Callable, config: Mapping[str, Any]) -> bool:
    if not config.get("inject_messages", False):
        return False

    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(getattr(tool, "impl", tool), agent_type)


def is_agent_tool_impl(impl: Any) -> bool:
    if getattr(impl, "is_agent_tool", False):
        return True
    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(impl, agent_type)
