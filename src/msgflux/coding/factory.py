"""Local coding-agent session factory used by the trusted service manifest."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from msgflux.coding.checkpoints import CodingCheckpointExtension
from msgflux.coding.config import (
    CodingAgentCatalog,
    CodingAgentOptions,
    CodingPermission,
    coding_permission_set,
    coding_tool_allowed,
    validate_coding_catalog,
    workspace_coding_permissions,
)
from msgflux.coding.prompts import system_prompt
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.service import AgentSession, ServiceThread
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import (
    AgentTool,
    ApplyPatchTool,
    BashTool,
    DeleteTool,
    EditTool,
    ReadFileTool,
    SendUserMessageTool,
    WebFetchTool,
    WebSearchTool,
    WriteTool,
)
from msgflux.tools.builtin.workspace_query import GlobTool, GrepTool, LsTool

_TOOL_FACTORIES: dict[str, Callable[[], Any]] = {
    "bash": BashTool,
    "read": ReadFileTool,
    "write": WriteTool,
    "edit": EditTool,
    "delete": DeleteTool,
    "apply_patch": ApplyPatchTool,
    "ls": LsTool,
    "glob": GlobTool,
    "grep": GrepTool,
    "web_fetch": WebFetchTool,
    "web_search": WebSearchTool,
}


def _tool_instances(
    active: tuple[str, ...],
    deferred: tuple[str, ...],
    *,
    owned_resources: list[Any],
    supports_vision: bool,
) -> list[Any]:
    tools: list[Any] = []
    for name, is_deferred in (
        *((name, False) for name in active),
        *((name, True) for name in deferred),
    ):
        instance = (
            ReadFileTool(supports_vision=supports_vision)
            if name == "read"
            else _TOOL_FACTORIES[name]()
        )
        if callable(getattr(instance, "aclose", None)) or callable(
            getattr(instance, "close", None)
        ):
            owned_resources.append(instance)
        else:
            owned_resources.extend(
                resource
                for resource in (
                    getattr(instance, "model", None),
                    getattr(instance, "retriever", None),
                )
                if resource is not None
            )
        config = dict(
            getattr(instance, "tool_config", getattr(type(instance), "tool_config", {}))
        )
        config["defer_loading"] = is_deferred
        if name == "bash":
            config["allow_background"] = True
        instance.tool_config = config
        tools.append(instance)
    return tools


def _load_prompt(
    options: CodingAgentOptions,
    *,
    working_directory: Path,
    config_path: Path,
    interactive: bool,
    permissions: tuple[CodingPermission, ...],
) -> str:
    custom_instructions = options.instructions
    if options.prompt_file is not None:
        try:
            custom_instructions = options.prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Unable to read prompt_file for agent {options.agent_id!r} "
                f"from {config_path}: {exc}"
            ) from exc
    return system_prompt(
        str(working_directory),
        agent_name=options.agent_id,
        interactive=interactive,
        permissions=permissions,
        custom_instructions=custom_instructions,
    )


async def _close_resources(
    resources: tuple[Any, ...], workspace: AgentWorkspace
) -> None:
    seen: set[int] = set()
    errors = []
    for resource in (*reversed(resources), workspace):
        if resource is None or id(resource) in seen:
            continue
        seen.add(id(resource))
        close = getattr(resource, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                errors.append(exc)
            continue
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise ExceptionGroup("Coding-agent resource cleanup failed", errors)


def create_coding_session_factory(
    *,
    options: CodingAgentOptions,
    catalog: CodingAgentCatalog,
    config_path: Path,
    runtime_dir: Path,
    accounts=None,
) -> Callable[[ServiceThread], Awaitable[AgentSession]]:
    """Create a lazy, per-thread AgentSession factory for a coding entry."""
    if not isinstance(options, CodingAgentOptions):
        raise TypeError("options must be CodingAgentOptions")
    if not isinstance(catalog, CodingAgentCatalog):
        raise TypeError("catalog must be CodingAgentCatalog")
    if options.agent_id not in catalog.agents:
        raise ValueError(f"Coding catalog lacks entry {options.agent_id!r}")
    validate_coding_catalog(catalog)
    config_path = Path(config_path).expanduser().resolve()
    runtime_dir = Path(runtime_dir).expanduser().resolve()
    builder = _CodingSessionBuilder(
        options=options,
        catalog=catalog,
        config_path=config_path,
        runtime_dir=runtime_dir,
        accounts=accounts,
    )
    return builder.create_session


class _CodingSessionBuilder:
    def __init__(
        self,
        *,
        options: CodingAgentOptions,
        catalog: CodingAgentCatalog,
        config_path: Path,
        runtime_dir: Path,
        accounts=None,
    ) -> None:
        self.options = options
        self.catalog = catalog
        self.config_path = config_path
        self.runtime_dir = runtime_dir
        self.accounts = accounts

    async def create_session(self, thread: ServiceThread) -> AgentSession:
        if not isinstance(thread, ServiceThread):
            raise TypeError("thread must be ServiceThread")
        working_directory = Path(thread.cwd or Path.cwd()).expanduser().resolve()
        registry = SQLiteWorkspaceRegistry(self.runtime_dir / "workspaces.sqlite3")
        try:
            workspace = await AgentWorkspace.open(
                LocalWorkspaceBackend(
                    working_directory, registry=registry, allow_processes=True
                ),
                f"coding-{sha256(thread.thread_id.encode()).hexdigest()}",
                permissions=coding_permission_set(self.options.permissions),
                requirements=SandboxRequirements(),
                write_guarantee="cooperative_compare",
            )
        except BaseException:
            registry.close()
            raise
        models: list[Any] = []
        tool_resources: list[Any] = []
        try:
            agent = self._build_agent(
                self.options,
                name=self.options.agent_id,
                workspace=workspace,
                working_directory=working_directory,
                models=models,
                tool_resources=tool_resources,
                interactive=thread.interactive,
            )
            return AgentSession(
                agent,
                on_close=lambda: _close_resources(
                    (*models, *tool_resources, registry), workspace
                ),
            )
        except BaseException as error:
            try:
                await _close_resources((*models, *tool_resources, registry), workspace)
            except Exception as cleanup_error:
                error.add_note(f"Resource cleanup failed: {cleanup_error}")
            raise

    def _build_agent(
        self,
        selected: CodingAgentOptions,
        *,
        name: str,
        workspace: AgentWorkspace,
        working_directory: Path,
        models: list[Any],
        tool_resources: list[Any],
        ancestors: frozenset[str] = frozenset(),
        interactive: bool = False,
    ) -> Agent:
        if selected.agent_id in ancestors:
            raise ValueError(f"Subagent cycle detected at {selected.agent_id!r}")
        workspace = workspace.with_permissions(
            coding_permission_set(selected.permissions)
        )
        permissions = workspace_coding_permissions(workspace.permissions)
        active = tuple(
            name
            for name in selected.tools_active
            if coding_tool_allowed(name, permissions)
        )
        deferred = tuple(
            name
            for name in selected.tools_deferred
            if coding_tool_allowed(name, permissions)
        )
        model = (
            _create_model(selected, accounts=self.accounts)
            if self.accounts is not None
            else _create_model(selected)
        )
        models.append(model)
        children = self._build_children(
            selected,
            workspace=workspace,
            working_directory=working_directory,
            models=models,
            tool_resources=tool_resources,
            ancestors=ancestors | {selected.agent_id},
        )
        tools = _tool_instances(
            active,
            deferred,
            owned_resources=tool_resources,
            supports_vision=_model_supports_vision(model),
        )
        if interactive and not _model_has_native_commentary(model):
            tools.append(SendUserMessageTool())
        if children:
            dispatcher = AgentTool()
            dispatcher.tool_config = {
                **dispatcher.tool_config,
                "allow_background": True,
            }
            tools = [dispatcher, *children, *tools]
        agent = Agent(
            name=name,
            description=selected.description,
            model=model,
            system_prompt=_load_prompt(
                selected,
                working_directory=working_directory,
                config_path=self.config_path,
                interactive=interactive,
                permissions=permissions,
            ),
            # Children return through AgentTool as checkpointed tool results, so
            # they must settle to a plain value before the parent records them.
            config={"stream": selected.agent_id == self.options.agent_id},
            tools=tools,
            workspace=workspace,
            agent_dir=(
                self.runtime_dir.parent
                if selected.agent_id == self.options.agent_id
                else None
            ),
        )
        agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
        agent.tool_library.extensions["background_tasks"].enable_grouping()
        agent.tool_library.apply_default_usage_guidance()
        return agent

    def _build_children(
        self,
        selected: CodingAgentOptions,
        *,
        workspace: AgentWorkspace,
        working_directory: Path,
        models: list[Any],
        tool_resources: list[Any],
        ancestors: frozenset[str],
    ) -> list[Agent]:
        children = []
        for child_id in selected.subagents:
            try:
                child_options = self.catalog.agents[child_id]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown subagent {child_id!r} referenced by {selected.agent_id!r}"
                ) from exc
            children.append(
                self._build_agent(
                    child_options,
                    name=child_options.agent_id,
                    workspace=workspace,
                    working_directory=working_directory,
                    models=models,
                    tool_resources=tool_resources,
                    ancestors=ancestors,
                )
            )
        return children


def _model_supports_vision(model: Any) -> bool:
    profile = getattr(model, "profile", None)
    modalities = getattr(profile, "modalities", None)
    input_modalities = getattr(modalities, "input", ())
    return "image" in input_modalities


def _model_has_native_commentary(model: Any) -> bool:
    capabilities = getattr(model, "api_mode_capabilities", None)
    if getattr(capabilities, "assistant_commentary", False):
        return True
    supports_unphased = getattr(model, "supports_unphased_tool_commentary", None)
    return bool(callable(supports_unphased) and supports_unphased())


def _create_model(options: CodingAgentOptions, *, accounts=None):
    from msgflux.models.model import Model  # noqa: PLC0415

    kwargs = {"reasoning_effort": options.reasoning_effort}
    if accounts is not None:
        provider = options.model.split("/", 1)[0]
        kwargs["credential_resolver"] = accounts.resolver(provider)
    return Model.chat_completion(options.model, **kwargs)


__all__ = [
    "CodingAgentCatalog",
    "CodingAgentOptions",
    "create_coding_session_factory",
]
