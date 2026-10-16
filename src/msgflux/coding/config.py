"""Lightweight resolved configuration contracts for coding agents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import msgspec

from msgflux.coding.presets import get_coding_preset
from msgflux.runtime.permissions import PermissionSet

_TOOL_SELECTORS = frozenset(
    {
        "workspace",
        "bash",
        "read",
        "write",
        "edit",
        "delete",
        "apply_patch",
        "ls",
        "glob",
        "grep",
        "web_fetch",
        "web_search",
    }
)
_OPENAI_WORKSPACE_TOOLS = ("bash", "read", "apply_patch")
_NON_OPENAI_WORKSPACE_TOOLS = ("bash", "read", "edit", "write", "delete")
CodingPermission = Literal["read", "write", "execute"]
DEFAULT_CODING_PERMISSIONS = ("read", "write", "execute")
_PERMISSION_GRANTS = {
    "read": ("filesystem.read", "filesystem.list"),
    "write": ("filesystem.write", "filesystem.delete", "filesystem.mkdir"),
    "execute": ("process.execute",),
}
_TOOL_PERMISSIONS = {
    "read": {"read"},
    "ls": {"read"},
    "glob": {"read"},
    "grep": {"read"},
    "write": {"read", "write"},
    "edit": {"read", "write"},
    "delete": {"read", "write"},
    "apply_patch": {"read", "write"},
    "bash": {"execute"},
}


def validate_coding_permissions(permissions: tuple[CodingPermission, ...]) -> None:
    if not isinstance(permissions, (tuple, list)):
        raise TypeError("permissions must be a list or tuple")
    if any(
        not isinstance(name, str) or name not in _PERMISSION_GRANTS
        for name in permissions
    ):
        raise ValueError("permissions must contain only read, write, or execute")
    if len(permissions) != len(set(permissions)):
        raise ValueError("permissions must not contain duplicates")


def coding_permission_set(permissions: tuple[CodingPermission, ...]) -> PermissionSet:
    """Map manifest permissions to existing workspace capabilities."""
    validate_coding_permissions(permissions)
    return PermissionSet(
        grant for name in permissions for grant in _PERMISSION_GRANTS[name]
    )


def workspace_coding_permissions(
    permissions: PermissionSet,
) -> tuple[CodingPermission, ...]:
    """Recover effective manifest-level capabilities after delegation clipping."""
    return tuple(
        name
        for name, grants in _PERMISSION_GRANTS.items()
        if permissions.allows(grants)
    )


def coding_tool_allowed(name: str, permissions: tuple[CodingPermission, ...]) -> bool:
    return _TOOL_PERMISSIONS.get(name, set()).issubset(permissions)


class CodingAgentOptions(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Resolved, immutable settings for one manifest coding agent."""

    agent_id: str
    model: str
    reasoning_effort: str
    tools_active: tuple[str, ...] = ()
    tools_deferred: tuple[str, ...] = ()
    prompt_file: Path | None = None
    subagents: tuple[str, ...] = ()
    permissions: tuple[CodingPermission, ...] = DEFAULT_CODING_PERMISSIONS
    description: str | None = None
    instructions: str | None = None
    extensions: tuple[str, ...] = ()


def coding_agent_options(
    agent_id: str,
    model: str,
    *,
    reasoning_effort: str = "medium",
    preset: str | None = None,
    description: str | None = None,
    permissions: tuple[CodingPermission, ...] | None = None,
    active: tuple[str, ...] | None = None,
    deferred: tuple[str, ...] | None = None,
    prompt_file: Path | None = None,
    subagents: tuple[str, ...] = (),
    extensions: tuple[str, ...] = (),
) -> CodingAgentOptions:
    """Resolve explicit settings over builtin defaults for Python or TOML callers."""
    defaults = get_coding_preset(preset) if preset is not None else None
    options = CodingAgentOptions(
        agent_id=agent_id,
        model=model,
        reasoning_effort=reasoning_effort,
        tools_active=active
        if active is not None
        else defaults.active
        if defaults
        else ("workspace",),
        tools_deferred=deferred
        if deferred is not None
        else defaults.deferred
        if defaults
        else (),
        permissions=permissions
        if permissions is not None
        else defaults.permissions
        if defaults
        else DEFAULT_CODING_PERMISSIONS,
        description=description
        if description is not None
        else defaults.description
        if defaults
        else None,
        instructions=defaults.instructions if defaults else None,
        prompt_file=prompt_file,
        subagents=subagents,
        extensions=extensions,
    )
    return resolve_coding_agent_options(options)


class CodingAgentCatalog(msgspec.Struct, frozen=True):
    """Resolved coding-agent entries available for child dispatch."""

    agents: Mapping[str, CodingAgentOptions]


def validate_model_path(model_path: str) -> None:
    """Validate `provider/model-id` syntax without loading a model provider."""
    if not isinstance(model_path, str) or any(
        character.isspace() for character in model_path
    ):
        raise ValueError("model must use the form 'provider/model-id'")
    provider, separator, model_id = model_path.partition("/")
    if not separator or not provider or not model_id:
        raise ValueError("model must use the form 'provider/model-id'")


def _openai_tool_family(model_path: str) -> bool:
    validate_model_path(model_path)
    provider, _, model_id = model_path.partition("/")
    return "openai" in provider.lower() or model_id.lower().startswith("openai/")


def _expand_selection(
    names: tuple[str, ...],
    *,
    model_path: str,
    option_name: str,
    permissions: tuple[CodingPermission, ...],
) -> tuple[str, ...]:
    selected: list[str] = []
    for name in names:
        if name not in _TOOL_SELECTORS:
            raise ValueError(f"Unknown {option_name} tool selector {name!r}")
        if name == "workspace":
            family = (
                _OPENAI_WORKSPACE_TOOLS
                if _openai_tool_family(model_path)
                else _NON_OPENAI_WORKSPACE_TOOLS
            )
            expanded = tuple(
                tool for tool in family if coding_tool_allowed(tool, permissions)
            )
            if "read" in permissions and "execute" not in permissions:
                expanded += ("ls", "glob", "grep")
        else:
            if not coding_tool_allowed(name, permissions):
                raise ValueError(
                    f"Tool {name!r} in {option_name} requires permissions: "
                    f"{', '.join(sorted(_TOOL_PERMISSIONS[name]))}"
                )
            expanded = (name,)
        for tool_name in expanded:
            if tool_name not in selected:
                selected.append(tool_name)
    return tuple(selected)


def resolve_coding_tools(
    options: CodingAgentOptions,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Expand workspace groups without importing tools."""
    if not isinstance(options, CodingAgentOptions):
        raise TypeError("options must be CodingAgentOptions")
    validate_model_path(options.model)
    _validate_extensions(options)
    validate_coding_permissions(options.permissions)
    active = _expand_selection(
        options.tools_active,
        model_path=options.model,
        option_name="active",
        permissions=options.permissions,
    )
    deferred = _expand_selection(
        options.tools_deferred,
        model_path=options.model,
        option_name="deferred",
        permissions=options.permissions,
    )
    return active, deferred


def resolve_coding_agent_options(
    options: CodingAgentOptions,
) -> CodingAgentOptions:
    """Return a copy with workspace groups replaced by concrete tool names."""
    active, deferred = resolve_coding_tools(options)
    return msgspec.structs.replace(
        options,
        tools_active=active,
        tools_deferred=deferred,
    )


def validate_coding_agent_options(options: CodingAgentOptions) -> None:
    """Validate one coding agent using the catalog-wide validation policy."""
    options = resolve_coding_agent_options(options)
    validate_coding_catalog(CodingAgentCatalog(agents={options.agent_id: options}))


def validate_coding_catalog(catalog: CodingAgentCatalog) -> None:
    """Validate resolved tool selections and subagent graph without side effects."""
    if not isinstance(catalog, CodingAgentCatalog):
        raise TypeError("catalog must be CodingAgentCatalog")

    for agent_id, options in catalog.agents.items():
        _validate_coding_agent(agent_id, options, catalog)

    _validate_subagent_graph(catalog)


def _validate_extensions(options: CodingAgentOptions) -> None:
    if not isinstance(options.extensions, (tuple, list)) or any(
        not isinstance(name, str) or not name.strip() for name in options.extensions
    ):
        raise ValueError(
            f"Agent {options.agent_id!r} extension names must be non-empty strings"
        )
    if len(options.extensions) != len(set(options.extensions)):
        raise ValueError(f"Agent {options.agent_id!r} has duplicate extension names")


def _validate_coding_agent(
    agent_id: str,
    options: CodingAgentOptions,
    catalog: CodingAgentCatalog,
) -> None:
    if not isinstance(options, CodingAgentOptions):
        raise TypeError(f"Coding catalog entry {agent_id!r} has invalid options")
    if agent_id != options.agent_id:
        raise ValueError(
            f"Coding catalog key {agent_id!r} does not match "
            f"agent_id {options.agent_id!r}"
        )
    _validate_extensions(options)
    validate_model_path(options.model)
    valid_ids = _TOOL_SELECTORS - {"workspace"}
    for category, names in (
        ("active", options.tools_active),
        ("deferred", options.tools_deferred),
    ):
        unknown = set(names) - valid_ids
        if unknown:
            raise ValueError(
                f"Resolved {category} tool IDs for {agent_id!r} are invalid: "
                f"{sorted(unknown)}"
            )
    overlap = set(options.tools_active) & set(options.tools_deferred)
    if overlap:
        raise ValueError(
            f"Tools for {agent_id!r} cannot be both active and deferred: "
            f"{', '.join(sorted(overlap))}"
        )
    validate_coding_permissions(options.permissions)
    for name in (*options.tools_active, *options.tools_deferred):
        if not coding_tool_allowed(name, options.permissions):
            raise ValueError(f"Agent {agent_id!r} lacks permissions for tool {name!r}")
    for child_id in options.subagents:
        if child_id not in catalog.agents:
            raise ValueError(
                f"Agent {agent_id!r} references unknown subagent {child_id!r}"
            )


def _validate_subagent_graph(catalog: CodingAgentCatalog) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(agent_id: str) -> None:
        if agent_id in visiting:
            raise ValueError(f"Subagent cycle detected at {agent_id!r}")
        if agent_id in visited:
            return
        visiting.add(agent_id)
        for child_id in catalog.agents[agent_id].subagents:
            visit(child_id)
        visiting.remove(agent_id)
        visited.add(agent_id)

    for agent_id in catalog.agents:
        visit(agent_id)


__all__ = [
    "CodingAgentCatalog",
    "CodingAgentOptions",
    "coding_agent_options",
    "resolve_coding_agent_options",
    "resolve_coding_tools",
    "validate_coding_agent_options",
    "validate_coding_catalog",
    "validate_model_path",
]
