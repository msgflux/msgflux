from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Callable

from msgflux.vulcano.extensions.api import ExtensionApi
from msgflux.vulcano.extensions.types import (
    EXTENSION_API_VERSION,
    EXTENSION_ENTRY_POINT_GROUP,
    ExtensionSettings,
    ExtensionSource,
)

__all__ = ["discover_extensions", "load_extension_candidate"]


@dataclass(frozen=True)
class ExtensionCandidate:
    source: ExtensionSource
    entry_point: metadata.EntryPoint | None = None

    @property
    def default_name(self) -> str:
        if self.entry_point is not None:
            return self.entry_point.name
        if self.source.path is None:
            return self.source.identifier
        if self.source.path.is_dir():
            return self.source.path.name
        return self.source.path.stem

    @property
    def identity(self) -> str:
        if self.source.path is not None:
            return f"path:{self.source.path}"
        return f"entry-point:{self.source.identifier}"


@dataclass(frozen=True)
class DiscoveryProblem:
    source: ExtensionSource
    message: str


@dataclass(frozen=True)
class DiscoveryResult:
    candidates: tuple[ExtensionCandidate, ...]
    problems: tuple[DiscoveryProblem, ...]


@dataclass(frozen=True)
class ExtensionDefinition:
    name: str
    api_version: int
    register: Callable[[ExtensionApi], object]
    module_names: tuple[str, ...] = ()


_PRIORITY = {
    "installed": 100,
    "user": 200,
    "project": 300,
    "cli": 400,
}


def discover_extensions(settings: ExtensionSettings) -> DiscoveryResult:
    if not settings.enabled:
        return DiscoveryResult((), ())

    candidates: list[ExtensionCandidate] = []
    problems: list[DiscoveryProblem] = []
    if settings.auto_discover:
        installed, installed_problems = _discover_entry_points()
        candidates.extend(installed)
        problems.extend(installed_problems)
        candidates.extend(
            _discover_directory(
                settings.resolved_user_directory,
                kind="user",
            )
        )
        if settings.trust_project:
            candidates.extend(
                _discover_directory(
                    settings.cwd / ".vulcano" / "extensions",
                    kind="project",
                )
            )

    for path in settings.explicit_paths:
        explicit, explicit_problems = _discover_explicit(path)
        candidates.extend(explicit)
        problems.extend(explicit_problems)

    deduplicated: dict[str, ExtensionCandidate] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (item.source.priority, item.source.identifier),
    ):
        deduplicated[candidate.identity] = candidate
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (item.source.priority, item.source.identifier),
    )
    return DiscoveryResult(tuple(ordered), tuple(problems))


def load_extension_candidate(
    candidate: ExtensionCandidate,
    generation: int,
) -> ExtensionDefinition:
    if candidate.entry_point is not None:
        target = candidate.entry_point.load()
        return _resolve_definition(target, candidate, ())

    path = candidate.source.path
    if path is None:
        raise RuntimeError("Local extension source has no path")
    module, module_names = _load_local_module(path, generation)
    try:
        return _resolve_definition(module, candidate, module_names)
    except Exception:
        _remove_modules(module_names)
        raise


def _discover_entry_points() -> tuple[
    list[ExtensionCandidate],
    list[DiscoveryProblem],
]:
    source = ExtensionSource(
        kind="installed",
        identifier=EXTENSION_ENTRY_POINT_GROUP,
        priority=_PRIORITY["installed"],
        trusted=True,
    )
    try:
        entry_points = metadata.entry_points(group=EXTENSION_ENTRY_POINT_GROUP)
    except Exception as error:
        return [], [DiscoveryProblem(source, str(error))]

    candidates: list[ExtensionCandidate] = []
    for entry_point in sorted(entry_points, key=lambda item: item.name):
        distribution = (
            entry_point.dist.name if entry_point.dist is not None else "unknown"
        )
        entry_source = ExtensionSource(
            kind="installed",
            identifier=f"{distribution}:{entry_point.name}",
            priority=_PRIORITY["installed"],
            trusted=True,
        )
        candidates.append(
            ExtensionCandidate(source=entry_source, entry_point=entry_point)
        )
    return candidates, []


def _discover_explicit(
    path: Path,
) -> tuple[list[ExtensionCandidate], list[DiscoveryProblem]]:
    resolved = path.expanduser().resolve()
    source = _local_source(resolved, kind="cli")
    if not resolved.exists():
        return [], [DiscoveryProblem(source, "Extension path does not exist")]
    if resolved.is_file():
        if resolved.suffix != ".py":
            return [], [DiscoveryProblem(source, "Extension file must end in .py")]
        return [ExtensionCandidate(source=source)], []
    if (resolved / "__init__.py").is_file():
        return [ExtensionCandidate(source=source)], []
    return _discover_directory(resolved, kind="cli"), []


def _discover_directory(directory: Path, *, kind: str) -> list[ExtensionCandidate]:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        return []

    candidates: list[ExtensionCandidate] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if child.name.startswith("_"):
            continue
        if child.is_file() and child.suffix == ".py":
            candidates.append(
                ExtensionCandidate(source=_local_source(child, kind=kind))
            )
        elif child.is_dir() and (child / "__init__.py").is_file():
            candidates.append(
                ExtensionCandidate(source=_local_source(child, kind=kind))
            )
    return candidates


def _local_source(path: Path, *, kind: str) -> ExtensionSource:
    resolved = path.expanduser().resolve()
    return ExtensionSource(
        kind=kind,
        identifier=str(resolved),
        path=resolved,
        priority=_PRIORITY[kind],
        trusted=True,
    )


def _load_local_module(
    path: Path, generation: int
) -> tuple[ModuleType, tuple[str, ...]]:
    entry_file = path / "__init__.py" if path.is_dir() else path
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module_name = f"_msgflux_vulcano_extension_{digest}_g{generation}"
    search_locations = [str(path)] if path.is_dir() else None
    spec = importlib.util.spec_from_file_location(
        module_name,
        entry_file,
        submodule_search_locations=search_locations,
    )
    if spec is None:
        raise ImportError(f"Unable to create module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        source = entry_file.read_text(encoding="utf-8")
        # Loading trusted Python source is the explicit purpose of this module.
        exec(compile(source, str(entry_file), "exec"), module.__dict__)  # noqa: S102
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, (module_name,)


def _resolve_definition(
    target: object,
    candidate: ExtensionCandidate,
    module_names: tuple[str, ...],
) -> ExtensionDefinition:
    if isinstance(target, ModuleType):
        name = getattr(target, "EXTENSION_NAME", candidate.default_name)
        api_version = getattr(target, "EXTENSION_API_VERSION", EXTENSION_API_VERSION)
        register = getattr(target, "setup", None) or getattr(target, "register", None)
    else:
        name = getattr(target, "name", candidate.default_name)
        api_version = getattr(target, "api_version", EXTENSION_API_VERSION)
        register = getattr(target, "register", None)
        if register is None and callable(target):
            register = target

    if not isinstance(name, str) or not name.strip():
        raise TypeError("Extension name must be a non-empty string")
    if not isinstance(api_version, int):
        raise TypeError("Extension api_version must be an integer")
    if not callable(register):
        raise TypeError("Extension must expose setup(api) or register(api)")
    return ExtensionDefinition(
        name=name.strip(),
        api_version=api_version,
        register=register,
        module_names=module_names,
    )


def _remove_modules(module_names: tuple[str, ...]) -> None:
    for module_name in reversed(module_names):
        loaded_names = tuple(
            name
            for name in sys.modules
            if name == module_name or name.startswith(f"{module_name}.")
        )
        for loaded_name in reversed(loaded_names):
            sys.modules.pop(loaded_name, None)
