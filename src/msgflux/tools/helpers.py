from importlib import import_module
from typing import Any, Callable, Collection, Mapping

BACKGROUND_TASK_TOOL_KIND = "background"
BACKGROUND_ACTIVITY_TOOL_KIND = "background_activity"
BACKGROUND_MESSAGE_TOOL_KIND = "background_message"
TOOL_BUCKET_KIND = "bucket"
RUNTIME_BACKGROUND_PARAM = "run_in_background"
BACKGROUND_TASK_CAPABILITIES = ("activity", "message")
DEFAULT_AGENT_BACKGROUND_CAPABILITIES = BACKGROUND_TASK_CAPABILITIES
RESERVED_TOOL_KINDS = {
    BACKGROUND_TASK_TOOL_KIND,
    BACKGROUND_ACTIVITY_TOOL_KIND,
    BACKGROUND_MESSAGE_TOOL_KIND,
}


def is_reserved_tool_kind(config: Mapping[str, Any]) -> bool:
    return config.get("tool_kind") in RESERVED_TOOL_KINDS


def is_background_capable(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("background", False) or config.get("allow_background", False)
    )


def normalize_background_capabilities(value: Collection[str]) -> tuple[str, ...]:
    if isinstance(value, (str, Mapping)) or not isinstance(value, Collection):
        raise TypeError("`background_capabilities` must be a collection of strings.")

    values = sorted(value) if isinstance(value, (set, frozenset)) else value
    capabilities = tuple(values)
    if not all(
        isinstance(capability, str) and capability for capability in capabilities
    ):
        raise ValueError("`background_capabilities` values must be non-empty strings.")
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("`background_capabilities` values must be unique.")

    unsupported = set(capabilities) - set(BACKGROUND_TASK_CAPABILITIES)
    if unsupported:
        names = ", ".join(sorted(f"`{name}`" for name in unsupported))
        expected = ", ".join(f"`{name}`" for name in BACKGROUND_TASK_CAPABILITIES)
        raise ValueError(
            f"Unsupported background capabilities: {names}. Expected one of {expected}."
        )
    return capabilities


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


def build_call_parameters_for_response(
    params: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if params is None:
        return None
    if hasattr(params, "to_dict"):
        parameters = params.to_dict()
    else:
        parameters = dict(params)
    for key in (
        "vars",
        "messages",
        "message",
        "task",
        "notification",
        "scope",
        "handle",
        "tool_call_id",
        RUNTIME_BACKGROUND_PARAM,
    ):
        parameters.pop(key, None)
    return parameters


def should_copy_injected_messages(tool: Callable, config: Mapping[str, Any]) -> bool:
    if not config.get("inject_messages", False):
        return False

    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(getattr(tool, "impl", tool), agent_type)


def is_agent_tool_impl(impl: Any) -> bool:
    agent_type = import_module("msgflux.nn.modules.agent").Agent
    agent_tool_type = import_module("msgflux.tools.builtin.agent_tool").AgentTool
    return isinstance(impl, (agent_type, agent_tool_type))
