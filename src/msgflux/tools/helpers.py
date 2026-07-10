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


def should_dispatch_background(
    config: Mapping[str, Any],
    call_params: dict[str, Any],
) -> bool:
    if config.get("background", False):
        call_params.pop(RUNTIME_BACKGROUND_PARAM, None)
        return True
    if not config.get("allow_background", False):
        return False
    return call_params.pop(RUNTIME_BACKGROUND_PARAM, False) is True


def coerce_tool_params(tool_name: str, tool_params: Any) -> dict[str, Any]:
    if tool_params is None:
        return {}
    if isinstance(tool_params, Mapping):
        return dict(tool_params)
    raise TypeError(
        f"Tool `{tool_name}` parameters must be a mapping or None, "
        f"given `{type(tool_params)}`."
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
