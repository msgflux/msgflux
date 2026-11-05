"""Lightweight resolved configuration contracts for coding agents."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import msgspec

from msgflux.coding.presets import get_coding_preset
from msgflux.runtime.permissions import PermissionSet
from msgflux.utils.validation import is_valid_account_name, validate_model_path

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


def _resolve_coding_agent_dir(runtime_dir: Path, agent_dir: Path | None = None) -> Path:
    """Resolve the coding state root without creating filesystem state."""
    selected_runtime = Path(runtime_dir).expanduser().resolve()
    if agent_dir is not None:
        # Keep the final path component intact so AgentResources can continue
        # rejecting a symlinked state root during its secure bind.
        return Path(os.path.abspath(Path(agent_dir).expanduser()))

    canonical_root = (Path.home() / ".msgflux").absolute()
    canonical_runtime = (canonical_root / "runtime").resolve()
    if selected_runtime == canonical_runtime:
        return canonical_root

    legacy_root = selected_runtime.parent
    raise ValueError(
        f"A non-default runtime_dir ({selected_runtime}) requires an explicit "
        f"agent_dir for coding-agent state. Existing threads from the previous "
        f"layout are at {legacy_root / 'threads'}; set defaults.agent_dir = "
        f"{str(legacy_root)!r} (or pass agent_dir={str(legacy_root)!r} to "
        "create_coding_session_factory) to resume them. No state is moved."
    )


def coding_tool_allowed(name: str, permissions: tuple[CodingPermission, ...]) -> bool:
    return _TOOL_PERMISSIONS.get(name, set()).issubset(permissions)


class CodingModelDeployment(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One named model deployment available to a coding agent.

    ``model_name`` is the configuration alias; ``model`` is the provider/model-id
    path. ``account_name`` is ``None`` when the deployment inherits the
    execution account. Validation is offline only: it never constructs a
    provider or loads credentials.
    """

    model_name: str
    model: str
    description: str
    reasoning_effort: str | None = None
    account_name: str | None = None


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
    preset: str | None = None
    models: tuple[CodingModelDeployment, ...] = ()
    model_fallback: bool = False
    # Original tool selectors for gateway agents, preserved so the selected
    # deployment's tools can be projected per request. ``None`` for ordinary
    # single-model agents, whose tools are resolved from ``tools_active`` /
    # ``tools_deferred`` directly.
    tools_active_selectors: tuple[str, ...] | None = None
    tools_deferred_selectors: tuple[str, ...] | None = None


def _validate_deployments(options: CodingAgentOptions) -> None:
    """Validate model deployments offline, without provider or credential use."""
    if not isinstance(options.model_fallback, bool):
        raise TypeError("model_fallback must be a bool")
    if not isinstance(options.models, (tuple, list)):
        raise TypeError("models must be a tuple of CodingModelDeployment")
    if options.model_fallback and not options.models:
        raise ValueError(f"Agent {options.agent_id!r} model_fallback requires models")
    seen: set[str] = set()
    for deployment in options.models:
        _validate_deployment(deployment, options.agent_id)
        if deployment.model_name in seen:
            raise ValueError(
                f"Agent {options.agent_id!r} has duplicate model alias "
                f"{deployment.model_name!r}"
            )
        seen.add(deployment.model_name)


def _validate_deployment(deployment: CodingModelDeployment, agent_id: str) -> None:
    if not isinstance(deployment, CodingModelDeployment):
        raise TypeError(
            f"Agent {agent_id!r} models must be CodingModelDeployment entries"
        )
    if (
        not isinstance(deployment.model_name, str)
        or not deployment.model_name.strip()
        or any(character.isspace() for character in deployment.model_name)
    ):
        raise ValueError(
            f"Agent {agent_id!r} model alias must be a non-empty "
            "string without whitespace"
        )
    try:
        validate_model_path(deployment.model)
    except ValueError as exc:
        raise ValueError(
            f"Agent {agent_id!r} model {deployment.model_name!r}: {exc}"
        ) from exc
    if not isinstance(deployment.description, str) or not (
        deployment.description.strip()
    ):
        raise ValueError(
            f"Agent {agent_id!r} model {deployment.model_name!r} "
            "description must be non-empty"
        )
    if deployment.reasoning_effort is not None and (
        not isinstance(deployment.reasoning_effort, str)
        or not deployment.reasoning_effort.strip()
    ):
        raise ValueError(
            f"Agent {agent_id!r} model {deployment.model_name!r} "
            "reasoning_effort must be non-empty when provided"
        )
    if deployment.account_name is not None and not is_valid_account_name(
        deployment.account_name
    ):
        raise ValueError(
            f"Agent {agent_id!r} model {deployment.model_name!r} "
            "account_name is invalid"
        )


def coding_agent_options(
    agent_id: str,
    model: str | None = None,
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
    models: tuple[CodingModelDeployment, ...]
    | list[CodingModelDeployment]
    | None = None,
    model_fallback: bool = False,
) -> CodingAgentOptions:
    """Resolve explicit settings over builtin defaults for Python or TOML callers.

    ``models`` and an explicit scalar ``model`` are mutually exclusive. With
    ``models``, the first deployment is the default and ``options.model`` is
    its provider/model-id path for backwards compatibility.
    """
    if models is not None:
        if not isinstance(models, (tuple, list)):
            raise TypeError("models must be a tuple of CodingModelDeployment")
        if not models:
            raise ValueError("models must not be empty when provided")
        if model is not None:
            raise ValueError(
                "model and models are mutually exclusive; pass either an "
                "explicit model or agent models, not both"
            )
        deployments = tuple(models)
        for deployment in deployments:
            _validate_deployment(deployment, agent_id)
        resolved_model = deployments[0].model
    else:
        if model is None:
            raise ValueError("model is required when models are not provided")
        if model_fallback:
            raise ValueError("model_fallback requires models")
        deployments = ()
        resolved_model = model
    defaults = get_coding_preset(preset) if preset is not None else None
    options = CodingAgentOptions(
        agent_id=agent_id,
        model=resolved_model,
        models=deployments,
        model_fallback=model_fallback,
        preset=preset,
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
                tool
                for tool in family
                if coding_tool_allowed(tool, permissions)
                and (tool != "delete" or "execute" not in permissions)
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
    _validate_preset(options)
    _validate_extensions(options)
    _validate_deployments(options)
    validate_coding_permissions(options.permissions)
    if not options.models:
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
    # Gateway: expand the original selectors per deployment and take the
    # union in first-occurrence order across deployments.
    active_selectors = (
        options.tools_active_selectors
        if options.tools_active_selectors is not None
        else options.tools_active
    )
    deferred_selectors = (
        options.tools_deferred_selectors
        if options.tools_deferred_selectors is not None
        else options.tools_deferred
    )
    active_union: list[str] = []
    deferred_union: list[str] = []
    for deployment in options.models:
        for tool_name in _expand_selection(
            active_selectors,
            model_path=deployment.model,
            option_name="active",
            permissions=options.permissions,
        ):
            if tool_name not in active_union:
                active_union.append(tool_name)
        for tool_name in _expand_selection(
            deferred_selectors,
            model_path=deployment.model,
            option_name="deferred",
            permissions=options.permissions,
        ):
            if tool_name not in deferred_union:
                deferred_union.append(tool_name)
    overlap = set(active_union) & set(deferred_union)
    if overlap:
        raise ValueError(
            f"Tools for {options.agent_id!r} cannot be both active and "
            f"deferred: {', '.join(sorted(overlap))}"
        )
    return tuple(active_union), tuple(deferred_union)


def resolve_coding_agent_options(
    options: CodingAgentOptions,
) -> CodingAgentOptions:
    """Return a copy with workspace groups replaced by concrete tool names.

    Gateway agents keep their original selectors in
    ``tools_active_selectors`` / ``tools_deferred_selectors`` while
    ``tools_active`` / ``tools_deferred`` hold the union across deployments.
    """
    active, deferred = resolve_coding_tools(options)
    if not options.models:
        return msgspec.structs.replace(
            options,
            tools_active=active,
            tools_deferred=deferred,
        )
    return msgspec.structs.replace(
        options,
        tools_active=active,
        tools_deferred=deferred,
        tools_active_selectors=(
            options.tools_active_selectors
            if options.tools_active_selectors is not None
            else tuple(options.tools_active)
        ),
        tools_deferred_selectors=(
            options.tools_deferred_selectors
            if options.tools_deferred_selectors is not None
            else tuple(options.tools_deferred)
        ),
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


def _validate_preset(options: CodingAgentOptions) -> None:
    if options.preset is not None:
        get_coding_preset(options.preset)


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
    _validate_preset(options)
    _validate_deployments(options)
    validate_model_path(options.model)
    _validate_original_selectors(options)
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


def _validate_original_selectors(options: CodingAgentOptions) -> None:
    agent_id = options.agent_id
    for category, selectors in (
        ("active", options.tools_active_selectors),
        ("deferred", options.tools_deferred_selectors),
    ):
        if selectors is None:
            continue
        unknown_selectors = set(selectors) - _TOOL_SELECTORS
        if unknown_selectors:
            raise ValueError(
                f"{category.capitalize()} tool selectors for {agent_id!r} are "
                f"invalid: {sorted(unknown_selectors)}"
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
    "CodingModelDeployment",
    "coding_agent_options",
    "resolve_coding_agent_options",
    "resolve_coding_tools",
    "validate_coding_agent_options",
    "validate_coding_catalog",
    "validate_model_path",
]
