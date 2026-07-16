from __future__ import annotations

import inspect
import itertools
import sys
import weakref
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from msgflux.vulcano.commands import CommandRegistry
from msgflux.vulcano.events import DomainEvent
from msgflux.vulcano.extensions.agent import AgentAdapter, _AgentBinding
from msgflux.vulcano.extensions.api import ExtensionApi
from msgflux.vulcano.extensions.loader import (
    ExtensionCandidate,
    discover_extensions,
    load_extension_candidate,
)
from msgflux.vulcano.extensions.types import (
    EXTENSION_API_VERSION,
    ExtensionContext,
    ExtensionDiagnostic,
    ExtensionInfo,
    ExtensionLoadReport,
    ExtensionObserver,
    ExtensionReloadReport,
    ExtensionSettings,
    ExtensionSource,
)

__all__ = ["ExtensionManager"]


@dataclass
class _LoadedExtension:
    info: ExtensionInfo
    api: ExtensionApi
    module_names: tuple[str, ...]


@dataclass(frozen=True)
class _Observer:
    identifier: int
    owner: str
    event_type: str
    handler: ExtensionObserver
    context: ExtensionContext


class _ObserverRegistration:
    def __init__(self, manager: ExtensionManager, identifier: int) -> None:
        self._manager_ref = weakref.ref(manager)
        self._identifier = identifier

    def remove(self) -> None:
        manager = self._manager_ref()
        if manager is not None:
            manager._observers.pop(self._identifier, None)


class _ExtensionControlFacade:
    def __init__(self, manager: ExtensionManager) -> None:
        self._manager_ref = weakref.ref(manager)

    @property
    def enabled(self) -> bool:
        return self._manager().enabled

    @property
    def records(self) -> tuple[ExtensionInfo, ...]:
        return self._manager().records

    @property
    def diagnostics(self) -> tuple[ExtensionDiagnostic, ...]:
        return self._manager().diagnostics

    async def reload(self) -> ExtensionReloadReport:
        return await self._manager().reload()

    def _manager(self) -> ExtensionManager:
        manager = self._manager_ref()
        if manager is None:
            raise RuntimeError("Extension manager is no longer available")
        return manager


class ExtensionManager:
    """Loads extensions and owns everything registered by each generation."""

    def __init__(
        self,
        commands: CommandRegistry,
        settings: ExtensionSettings,
        *,
        services: Mapping[str, object] | None = None,
        agent: object | None = None,
        agent_adapter: AgentAdapter | None = None,
    ) -> None:
        self.commands = commands
        self.settings = settings
        self._generation = 0
        self._records: list[ExtensionInfo] = []
        self._diagnostics: list[ExtensionDiagnostic] = []
        self._loaded: dict[str, _LoadedExtension] = {}
        self._observers: dict[int, _Observer] = {}
        self._observer_ids = itertools.count()
        self._agent_binding = _AgentBinding(agent, agent_adapter)
        control = _ExtensionControlFacade(self)
        service_values = dict(services or {})
        if "extensions" in service_values:
            raise ValueError("The 'extensions' runtime service name is reserved")
        service_values["extensions"] = control
        self._services = MappingProxyType(service_values)
        core_source = ExtensionSource(
            kind="core",
            identifier="vulcano",
            priority=0,
            trusted=True,
        )
        core_context = ExtensionContext(
            cwd=settings.cwd,
            generation=0,
            source=core_source,
            services=self._services,
        )
        self._api = ExtensionApi(
            self,
            owner="vulcano",
            context=core_context,
            agent_binding=self._agent_binding,
        )

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def records(self) -> tuple[ExtensionInfo, ...]:
        return tuple(self._records)

    @property
    def diagnostics(self) -> tuple[ExtensionDiagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def services(self) -> Mapping[str, object]:
        return self._services

    @property
    def api(self) -> ExtensionApi:
        return self._api

    def bind_agent(
        self,
        agent: object,
        adapter: AgentAdapter | None = None,
    ) -> None:
        self._agent_binding.bind(agent, adapter)

    async def load_all(self) -> ExtensionLoadReport:
        self._records = []
        self._diagnostics = []
        if not self.enabled:
            return ExtensionLoadReport()

        self._generation += 1
        discovery = discover_extensions(self.settings)
        failed: list[ExtensionInfo] = []
        for problem in discovery.problems:
            info = ExtensionInfo(
                name=problem.source.identifier,
                api_version=0,
                source=problem.source,
                generation=self._generation,
                state="failed",
                error=problem.message,
            )
            diagnostic = ExtensionDiagnostic(
                extension=info.name,
                phase="discover",
                message=problem.message,
                source=problem.source,
            )
            failed.append(info)
            self._records.append(info)
            self._diagnostics.append(diagnostic)

        loaded: list[ExtensionInfo] = []
        for candidate in discovery.candidates:
            info, diagnostic = await self._load_one(candidate)
            self._records.append(info)
            if diagnostic is None:
                loaded.append(info)
            else:
                failed.append(info)
                self._diagnostics.append(diagnostic)
        return ExtensionLoadReport(
            loaded=tuple(loaded),
            failed=tuple(failed),
            diagnostics=tuple(self._diagnostics),
        )

    async def reload(self) -> ExtensionReloadReport:
        unloaded, unload_diagnostics = await self.unload_all()
        report = await self.load_all()
        diagnostics = (*unload_diagnostics, *report.diagnostics)
        self._diagnostics = list(diagnostics)
        return ExtensionReloadReport(
            unloaded=unloaded,
            loaded=report.loaded,
            failed=report.failed,
            diagnostics=diagnostics,
        )

    async def unload_all(
        self,
    ) -> tuple[tuple[ExtensionInfo, ...], tuple[ExtensionDiagnostic, ...]]:
        unloaded: list[ExtensionInfo] = []
        diagnostics: list[ExtensionDiagnostic] = []
        for loaded in reversed(tuple(self._loaded.values())):
            cleanup_errors = await loaded.api._deactivate()
            self._remove_modules(loaded.module_names)
            unloaded.append(loaded.info)
            for message in cleanup_errors:
                diagnostics.append(
                    ExtensionDiagnostic(
                        extension=loaded.info.name,
                        phase="unload",
                        message=message,
                        source=loaded.info.source,
                    )
                )
        self._loaded.clear()
        self._observers.clear()
        self._records = []
        self._diagnostics = list(diagnostics)
        return tuple(unloaded), tuple(diagnostics)

    def register_observer(
        self,
        owner: str,
        event_type: str,
        observer: ExtensionObserver,
        context: ExtensionContext,
    ) -> _ObserverRegistration:
        if not event_type:
            raise ValueError("Observer event type cannot be empty")
        identifier = next(self._observer_ids)
        self._observers[identifier] = _Observer(
            identifier=identifier,
            owner=owner,
            event_type=event_type,
            handler=observer,
            context=context,
        )
        return _ObserverRegistration(self, identifier)

    async def notify(self, event: DomainEvent) -> tuple[ExtensionDiagnostic, ...]:
        diagnostics: list[ExtensionDiagnostic] = []
        observers = sorted(self._observers.values(), key=lambda item: item.identifier)
        for observer in observers:
            if observer.event_type not in {event.type, "*"}:
                continue
            try:
                result = observer.handler(event, observer.context)
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                diagnostic = ExtensionDiagnostic(
                    extension=observer.owner,
                    phase="observe",
                    message=str(error),
                    source=observer.context.source,
                )
                diagnostics.append(diagnostic)
                self._diagnostics.append(diagnostic)
        return tuple(diagnostics)

    async def _load_one(
        self,
        candidate: ExtensionCandidate,
    ) -> tuple[ExtensionInfo, ExtensionDiagnostic | None]:
        name = candidate.default_name
        api_version = 0
        api: ExtensionApi | None = None
        module_names: tuple[str, ...] = ()
        try:
            definition = load_extension_candidate(candidate, self._generation)
            name = definition.name
            api_version = definition.api_version
            module_names = definition.module_names
            if api_version != EXTENSION_API_VERSION:
                raise RuntimeError(
                    f"Unsupported extension API version {api_version}; "
                    f"expected {EXTENSION_API_VERSION}"
                )
            if name in self._loaded:
                raise RuntimeError(f"Extension name is already loaded: {name}")

            context = ExtensionContext(
                cwd=self.settings.cwd,
                generation=self._generation,
                source=candidate.source,
                services=self._services,
            )
            api = ExtensionApi(
                self,
                owner=name,
                context=context,
                agent_binding=self._agent_binding,
            )
            result = definition.register(api)
            if inspect.isawaitable(result):
                await result
            info = ExtensionInfo(
                name=name,
                api_version=api_version,
                source=candidate.source,
                generation=self._generation,
                state="loaded",
            )
            self._loaded[name] = _LoadedExtension(info, api, module_names)
            return info, None
        except Exception as error:
            cleanup_errors = await api._deactivate() if api is not None else ()
            self._remove_modules(module_names)
            message = str(error)
            if cleanup_errors:
                message = f"{message}; {'; '.join(cleanup_errors)}"
            info = ExtensionInfo(
                name=name,
                api_version=api_version,
                source=candidate.source,
                generation=self._generation,
                state="failed",
                error=message,
            )
            diagnostic = ExtensionDiagnostic(
                extension=name,
                phase="setup",
                message=message,
                source=candidate.source,
            )
            return info, diagnostic

    def _remove_modules(self, module_names: tuple[str, ...]) -> None:
        for module_name in reversed(module_names):
            loaded_names = tuple(
                name
                for name in sys.modules
                if name == module_name or name.startswith(f"{module_name}.")
            )
            for loaded_name in reversed(loaded_names):
                sys.modules.pop(loaded_name, None)
