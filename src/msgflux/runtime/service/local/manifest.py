"""Typed service agent configuration and default config initialization."""

from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import msgspec

from msgflux.coding.config import (
    DEFAULT_CODING_PERMISSIONS,
    CodingAgentCatalog,
    CodingAgentOptions,
    CodingPermission,
    resolve_coding_agent_options,
    validate_coding_catalog,
    validate_model_path,
)
from msgflux.runtime.service.store import validate_identifier

_CONFIG_TEMPLATE = """[defaults]
agent = "coding"
# Choose a model available to your configured provider, for example:
# model = "provider/model-id"
# reasoning_effort = "medium"

[agents.coding]
harness = "coding"
subagents = ["lite-coding"]

[agents.lite-coding]
harness = "coding"

# Example of a read-only explorer subagent:
# [agents.explorer]
# harness = "coding"
# permissions = ["read"]
"""


class _DefaultsConfig(msgspec.Struct, forbid_unknown_fields=True):
    agent: str
    model: str | None = None
    reasoning_effort: str | None = None


class _ToolsConfig(msgspec.Struct, forbid_unknown_fields=True):
    active: list[str] = msgspec.field(default_factory=list)
    deferred: list[str] = msgspec.field(default_factory=list)


class _AgentConfig(msgspec.Struct, forbid_unknown_fields=True):
    harness: str | None = None
    factory: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    prompt_file: str | None = None
    permissions: list[CodingPermission] | None = None
    subagents: list[str] = msgspec.field(default_factory=list)
    tools: _ToolsConfig | None = None


class _ServiceConfig(msgspec.Struct, forbid_unknown_fields=True):
    defaults: _DefaultsConfig
    agents: dict[str, _AgentConfig]


class ServiceManifest(msgspec.Struct, frozen=True):
    path: Path
    digest: str
    defaults: _DefaultsConfig
    agents: dict[str, _AgentConfig]
    coding_catalog: CodingAgentCatalog


def default_service_config_path() -> Path:
    return Path.home() / ".msgflux" / "config.toml"


def _template_with_model(model: str | None) -> str:
    if model is None:
        return _CONFIG_TEMPLATE
    try:
        validate_model_path(model)
    except ValueError as exc:
        raise ValueError(f"Invalid --model: {exc}") from exc
    quoted_model = msgspec.json.encode(model).decode("utf-8")
    return _CONFIG_TEMPLATE.replace(
        '# model = "provider/model-id"', f"model = {quoted_model}"
    )


def initialize_service_config(path: Path, *, model: str | None = None) -> Path:
    """Create the starter config once, publishing it atomically without overwrite."""
    template = _template_with_model(model)
    target = Path(path).expanduser().absolute()
    parent = target.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if parent == default_service_config_path().parent:
        metadata = parent.lstat()
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError("~/.msgflux must be a real directory")
        if metadata.st_uid != os.getuid():
            raise PermissionError("~/.msgflux must be owned by the current user")
        if metadata.st_mode & 0o077:
            parent.chmod(0o700)
    if target.is_dir() and not target.is_symlink():
        raise IsADirectoryError(target)
    if target.is_symlink() and not target.exists():
        raise FileNotFoundError(
            f"Config path is a dangling symlink: {target}; "
            "initialize its target explicitly"
        )
    if target.exists() or target.is_symlink():
        return target.resolve()

    descriptor, temporary_name = tempfile.mkstemp(prefix=".config-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(template)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            # A concurrent initializer won. Its contents are never replaced.
            pass
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target.resolve(strict=True)


def validate_factory_path(path: str) -> None:
    module_name, separator, attribute = path.partition(":")
    if (
        not separator
        or not module_name
        or not attribute
        or any(character.isspace() for character in path)
    ):
        raise ValueError("factory must use the form 'module:attribute'")


def load_factory(path: str) -> Any:
    validate_factory_path(path)
    module_name, _, attribute = path.partition(":")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("configured factory attribute must be callable")
    return factory


def load_service_manifest(path: Path, *, cwd: Path | None = None) -> ServiceManifest:
    """Parse and resolve the config without importing user factories."""
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (Path(cwd) if cwd is not None else Path.cwd()) / manifest_path
    manifest_path = manifest_path.resolve(strict=True)
    raw = manifest_path.read_bytes()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
        config = msgspec.convert(document, type=_ServiceConfig, strict=True)
    except (
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
        msgspec.ValidationError,
    ) as exc:
        raise ValueError(f"Invalid service config {manifest_path}: {exc}") from exc

    if not config.agents:
        raise ValueError("Service config must declare at least one agent")
    validate_identifier(config.defaults.agent, "default agent")
    if config.defaults.model is not None and not config.defaults.model.strip():
        raise ValueError("defaults.model must be non-empty when provided")
    if (
        config.defaults.reasoning_effort is not None
        and not config.defaults.reasoning_effort.strip()
    ):
        raise ValueError("defaults.reasoning_effort must be non-empty when provided")
    if config.defaults.agent not in config.agents:
        raise ValueError("defaults.agent must reference a declared agent")

    for agent_id, agent in config.agents.items():
        validate_identifier(agent_id, "agent_id")
        _validate_agent(agent_id, agent, config.agents, manifest_path.parent)
    coding_options = _resolve_coding_options(
        config, manifest_path.parent, manifest_path
    )
    coding_catalog = CodingAgentCatalog(agents=coding_options)
    validate_coding_catalog(coding_catalog)

    return ServiceManifest(
        path=manifest_path,
        digest=hashlib.sha256(raw).hexdigest(),
        defaults=config.defaults,
        agents=config.agents,
        coding_catalog=coding_catalog,
    )


def _resolve_coding_options(
    config: _ServiceConfig, config_dir: Path, config_path: Path
) -> dict[str, CodingAgentOptions]:
    options_by_id = {}
    for agent_id, agent in config.agents.items():
        if agent.harness != "coding":
            continue
        model = agent.model or config.defaults.model
        if model is None:
            raise ValueError(
                f"Coding agent {agent_id!r} in {config_path} requires "
                f"a model; set defaults.model or agents.{agent_id}.model "
                "to a provider/model-id"
            )
        prompt_file = Path(agent.prompt_file) if agent.prompt_file is not None else None
        if prompt_file is not None and not prompt_file.is_absolute():
            prompt_file = config_dir / prompt_file
        options = CodingAgentOptions(
            agent_id=agent_id,
            model=model,
            reasoning_effort=(
                agent.reasoning_effort or config.defaults.reasoning_effort or "medium"
            ),
            tools_active=tuple(
                agent.tools.active if agent.tools is not None else ("workspace",)
            ),
            tools_deferred=(
                tuple(agent.tools.deferred) if agent.tools is not None else ()
            ),
            prompt_file=prompt_file,
            subagents=tuple(agent.subagents),
            permissions=(
                tuple(agent.permissions)
                if agent.permissions is not None
                else DEFAULT_CODING_PERMISSIONS
            ),
        )
        options_by_id[agent_id] = resolve_coding_agent_options(options)
    return options_by_id


def _validate_agent(
    agent_id: str,
    agent: _AgentConfig,
    agents: dict[str, _AgentConfig],
    config_dir: Path,
) -> None:
    coding = agent.harness is not None
    custom = agent.factory is not None
    if coding == custom:
        raise ValueError(
            f"Agent {agent_id!r} must declare exactly one harness or factory"
        )

    if custom:
        validate_factory_path(agent.factory)
        if (
            any(
                value is not None
                for value in (
                    agent.model,
                    agent.reasoning_effort,
                    agent.prompt_file,
                    agent.tools,
                )
            )
            or agent.subagents
            or agent.permissions is not None
        ):
            raise ValueError(f"Custom agent {agent_id!r} cannot use coding options")
        return

    if agent.harness != "coding":
        raise ValueError(
            f"Agent {agent_id!r} uses unsupported harness {agent.harness!r}"
        )
    _validate_coding_agent(agent_id, agent, agents, config_dir)


def _validate_coding_agent(
    agent_id: str,
    agent: _AgentConfig,
    agents: dict[str, _AgentConfig],
    config_dir: Path,
) -> None:
    for label, value in (
        ("model", agent.model),
        ("reasoning_effort", agent.reasoning_effort),
    ):
        if value is not None and not value.strip():
            raise ValueError(f"Agent {agent_id!r} {label} must be non-empty")
    _resolve_prompt_file(agent_id, agent, config_dir)
    _validate_subagents(agent_id, agent, agents)


def _resolve_prompt_file(agent_id: str, agent: _AgentConfig, config_dir: Path) -> None:
    if agent.prompt_file is None:
        return
    if not agent.prompt_file.strip():
        raise ValueError(f"Agent {agent_id!r} prompt_file must be non-empty")
    prompt_path = Path(agent.prompt_file).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = config_dir / prompt_path
    try:
        prompt_path = prompt_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            f"Agent {agent_id!r} prompt_file could not be resolved: {exc}"
        ) from exc
    if not prompt_path.is_file():
        raise ValueError(f"Agent {agent_id!r} prompt_file must name a file")
    try:
        prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Agent {agent_id!r} prompt_file must be readable UTF-8 text"
        ) from exc
    agent.prompt_file = str(prompt_path)


def _validate_subagents(
    agent_id: str,
    agent: _AgentConfig,
    agents: dict[str, _AgentConfig],
) -> None:
    if len(agent.subagents) != len(set(agent.subagents)):
        raise ValueError(f"Agent {agent_id!r} has duplicate subagent IDs")
    for subagent in agent.subagents:
        validate_identifier(subagent, "subagent ID")
        if subagent not in agents:
            raise ValueError(
                f"Agent {agent_id!r} references unknown subagent {subagent!r}"
            )
        if subagent == agent_id:
            raise ValueError(f"Agent {agent_id!r} cannot be its own subagent")
        if agents[subagent].harness != "coding":
            raise ValueError(f"Subagent {subagent!r} must use the coding harness")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
