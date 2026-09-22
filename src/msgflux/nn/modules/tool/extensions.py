"""Composable policies, context providers, and dispatch strategies."""

from __future__ import annotations

import asyncio
import weakref
from copy import deepcopy
from typing import Any, Collection, Mapping

import msgspec

import msgflux.nn.functional as F
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool.definitions import (
    ContextBinding,
    DispatchSpec,
    ToolDefinition,
)
from msgflux.nn.modules.tool.execution import (
    AfterToolPolicy,
    BeforeDispatchPolicy,
    BeforeToolPolicy,
    DispatchRequest,
    ToolRuntimeContext,
)
from msgflux.tools.helpers import RUNTIME_BACKGROUND_PARAM
from msgflux.tools.runtime import ToolIntent, ToolOutcome, _require_name


class ToolExtension(Module):
    """Owned runtime capability registered by the future ToolLibrary."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = _require_name(name, "extension.name")
        self._registry_ref = None

    @property
    def registry(self) -> ToolExtensionRegistry:
        registry = self._registry_ref() if self._registry_ref is not None else None
        if registry is None:
            raise RuntimeError("The extension is not registered")
        return registry

    def _bind_registry(self, registry: ToolExtensionRegistry) -> None:
        self._registry_ref = weakref.ref(registry)

    def _unbind_registry(self) -> None:
        self._registry_ref = None

    def on_register(self, registry: ToolExtensionRegistry) -> None:
        """Run synchronous setup after the extension becomes visible."""

    async def aon_register(self, registry: ToolExtensionRegistry) -> None:
        """Run async setup; override this for network-backed extensions."""
        self.on_register(registry)

    def on_remove(self, registry: ToolExtensionRegistry) -> None:
        """Clean up synchronously owned resources."""

    async def aon_remove(self, registry: ToolExtensionRegistry) -> None:
        self.on_remove(registry)

    def __getstate__(self):
        state = super().__getstate__()
        state["_registry_ref"] = None
        return state


class ToolDispatch(ToolExtension):
    """Extension that owns one open dispatch name."""

    def __init__(self, name: str, *, dispatch_name: str) -> None:
        super().__init__(name)
        self.dispatch_name = _require_name(dispatch_name, "dispatch_name")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        raise NotImplementedError


class ToolPolicy(ToolExtension):
    """Sequential policy extension for tool planning and outcomes."""

    async def before_tool(
        self,
        payload: BeforeToolPolicy,
    ) -> BeforeToolPolicy | ToolOutcome:
        return payload

    async def before_dispatch(
        self,
        payload: BeforeDispatchPolicy,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        return payload

    async def after_tool(self, payload: AfterToolPolicy) -> AfterToolPolicy:
        return payload


class ForegroundDispatch(ToolDispatch):
    """Execute and await the tool inside the current run."""

    def __init__(self) -> None:
        super().__init__("dispatch_foreground", dispatch_name="foreground")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        return await request.execute(None)


class DetachedDispatch(ToolDispatch):
    """Start execution without retaining a task result."""

    def __init__(self) -> None:
        super().__init__("dispatch_detached", dispatch_name="detached")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        if request.context.get("sync_dispatch", False):
            F.detached(F.wait_for, request.execute, None)
        else:
            await F.adetached(request.execute, None)
        return ToolOutcome.dispatched(
            request.plan.intent,
            metadata={"dispatch": self.dispatch_name},
        )


class BackgroundDispatch(ToolDispatch):
    """Delegate durable scheduling to a runtime-provided background service."""

    def __init__(self) -> None:
        super().__init__("dispatch_background", dispatch_name="background")

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        scheduler = request.context.require("background_dispatcher")
        if hasattr(scheduler, "adispatch"):
            outcome = await scheduler.adispatch(request)
        elif hasattr(scheduler, "dispatch"):
            outcome = await asyncio.to_thread(scheduler.dispatch, request)
        else:
            raise TypeError(
                "`background_dispatcher` must define `dispatch` or `adispatch`"
            )
        if not isinstance(outcome, ToolOutcome):
            raise TypeError("Background dispatchers must return ToolOutcome")
        return outcome


class OptionalBackgroundDispatch(ToolDispatch):
    """Choose foreground or background from one reserved model argument."""

    def __init__(self) -> None:
        super().__init__(
            "dispatch_optional_background",
            dispatch_name="optional_background",
        )

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        argument = request.plan.dispatch.options.get(
            "argument",
            RUNTIME_BACKGROUND_PARAM,
        )
        if not isinstance(argument, str) or not argument:
            raise ValueError("Optional background dispatch requires an argument name")
        visible_arguments = dict(request.plan.visible_arguments)
        run_in_background = visible_arguments.pop(argument, False)
        if run_in_background not in {True, False, None}:
            raise TypeError(f"`{argument}` must be a bool or None")
        dispatch_name = "background" if run_in_background is True else "foreground"
        selected_plan = msgspec.structs.replace(
            request.plan,
            visible_arguments=visible_arguments,
            dispatch=DispatchSpec(name=dispatch_name),
        )
        dispatcher = self.registry.get_dispatch(dispatch_name)

        async def execute() -> ToolOutcome:
            return await request.execute(selected_plan)

        return await dispatcher.dispatch(
            DispatchRequest(
                plan=selected_plan,
                context=request.context,
                execute=lambda _plan=None: execute(),
            )
        )


class ContextRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Input delivered to a context provider extension."""

    binding: ContextBinding
    definition: ToolDefinition
    intent: ToolIntent
    context: ToolRuntimeContext


class ToolContextProvider(ToolExtension):
    """Extension that resolves one or more named runtime context sources."""

    def __init__(self, name: str, *, sources: Collection[str]) -> None:
        super().__init__(name)
        if isinstance(sources, str) or not isinstance(sources, Collection):
            raise TypeError("`sources` must be a collection of strings")
        normalized = tuple(
            _require_name(source, "context source") for source in sources
        )
        if not normalized:
            raise ValueError("`sources` must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("`sources` values must be unique")
        self.sources = normalized

    async def resolve(self, request: ContextRequest) -> Any:
        raise NotImplementedError


class RuntimeContextProvider(ToolContextProvider):
    """Resolve explicitly bound values from ToolRuntimeContext."""

    DEFAULT_SOURCES = (
        "handle",
        "message",
        "messages",
        "vars",
        "filesystem",
        "environment",
    )

    def __init__(self, sources: Collection[str] = DEFAULT_SOURCES) -> None:
        super().__init__("context_runtime", sources=sources)

    @staticmethod
    def _select_key(binding: ContextBinding, value: Any) -> Any:
        selected_key = binding.options.get("key")
        if selected_key is None:
            return value
        if not isinstance(selected_key, str) or not selected_key:
            raise TypeError("Context binding `key` must be a non-empty string")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Context source `{binding.source}` does not support key lookup"
            )
        if selected_key not in value:
            raise KeyError(
                f"Context source `{binding.source}` is missing `{selected_key}`"
            )
        return value[selected_key]

    @staticmethod
    def _select_keys(binding: ContextBinding, value: Any) -> Any:
        selected = binding.options.get("select")
        if selected is None:
            return value
        if isinstance(selected, str) or not isinstance(selected, Collection):
            raise TypeError("Context binding `select` must be a collection")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Context source `{binding.source}` does not support selection"
            )
        missing = [key for key in selected if key not in value]
        if missing:
            formatted = ", ".join(f"`{key}`" for key in missing)
            raise KeyError(f"Context source `{binding.source}` is missing {formatted}")
        return {key: value[key] for key in selected}

    async def resolve(self, request: ContextRequest) -> Any:
        binding = request.binding
        if binding.source not in request.context.values:
            if binding.required:
                raise RuntimeError(
                    f"Runtime context value `{binding.source}` is unavailable"
                )
            return _MISSING_CONTEXT

        value = request.context.values[binding.source]
        value = self._select_key(binding, value)
        value = self._select_keys(binding, value)
        if binding.options.get("copy", False):
            value = deepcopy(value)
        return value


class ToolExtensionHandle:
    """Ownership handle for one registered V2 extension."""

    def __init__(self, registry: ToolExtensionRegistry, name: str) -> None:
        self._registry_ref = weakref.ref(registry)
        self.name = name

    @property
    def active(self) -> bool:
        registry = self._registry_ref()
        return registry is not None and registry.has(self.name)

    def remove(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            registry.remove(self.name)

    async def aremove(self) -> None:
        registry = self._registry_ref()
        if registry is not None:
            await registry.aremove(self.name)


class ToolExtensionRegistry(Module):
    """Transactional ownership and capability indexes for Tool extensions."""

    def __init__(
        self,
        extensions: Collection[ToolExtension] = (),
        *,
        install_defaults: bool = False,
    ) -> None:
        super().__init__()
        self.extensions = ModuleDict()
        self._dispatches: dict[str, str] = {}
        self._context_sources: dict[str, str] = {}
        self._policies: list[str] = []
        initial = list(extensions)
        if install_defaults:
            initial = [
                ForegroundDispatch(),
                BackgroundDispatch(),
                DetachedDispatch(),
                OptionalBackgroundDispatch(),
                RuntimeContextProvider(),
                *initial,
            ]
        for extension in initial:
            self.register(extension)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        for extension in self.extensions.values():
            if isinstance(extension, ToolExtension):
                extension._bind_registry(self)

    def _validate_registration(
        self,
        extension: ToolExtension,
    ) -> tuple[str | None, tuple[str, ...]]:
        if not isinstance(extension, ToolExtension):
            raise TypeError("`extension` must be a ToolExtension")
        current_registry = (
            extension._registry_ref() if extension._registry_ref is not None else None
        )
        if current_registry is not None:
            raise ValueError(
                f"Extension `{extension.name}` is already registered on a registry"
            )
        if extension.name in self.extensions:
            raise ValueError(f"Extension `{extension.name}` is already registered")

        dispatch_name = (
            extension.dispatch_name if isinstance(extension, ToolDispatch) else None
        )
        if dispatch_name is not None and dispatch_name in self._dispatches:
            raise ValueError(f"Dispatch `{dispatch_name}` is already registered")

        context_sources = (
            extension.sources if isinstance(extension, ToolContextProvider) else ()
        )
        conflicts = [
            source for source in context_sources if source in self._context_sources
        ]
        if conflicts:
            formatted = ", ".join(f"`{source}`" for source in conflicts)
            raise ValueError(f"Context sources already registered: {formatted}")
        return dispatch_name, context_sources

    def _install(
        self,
        extension: ToolExtension,
    ) -> tuple[str | None, tuple[str, ...]]:
        dispatch_name, context_sources = self._validate_registration(extension)
        extension._bind_registry(self)
        self.extensions[extension.name] = extension
        if dispatch_name is not None:
            self._dispatches[dispatch_name] = extension.name
        for source in context_sources:
            self._context_sources[source] = extension.name
        if isinstance(extension, ToolPolicy):
            self._policies.append(extension.name)
        return dispatch_name, context_sources

    def _rollback_registration(
        self,
        extension: ToolExtension,
        dispatch_name: str | None,
        context_sources: tuple[str, ...],
    ) -> None:
        if extension.name in self.extensions:
            del self.extensions[extension.name]
        if dispatch_name is not None:
            self._dispatches.pop(dispatch_name, None)
        for source in context_sources:
            self._context_sources.pop(source, None)
        if extension.name in self._policies:
            self._policies.remove(extension.name)
        extension._unbind_registry()

    def register(self, extension: ToolExtension) -> ToolExtensionHandle:
        dispatch_name, context_sources = self._install(extension)
        try:
            extension.on_register(self)
        except Exception:
            try:
                extension.on_remove(self)
            finally:
                self._rollback_registration(
                    extension,
                    dispatch_name,
                    context_sources,
                )
            raise
        return ToolExtensionHandle(self, extension.name)

    async def aregister(
        self,
        extension: ToolExtension,
    ) -> ToolExtensionHandle:
        dispatch_name, context_sources = self._install(extension)
        try:
            await extension.aon_register(self)
        except Exception:
            try:
                await extension.aon_remove(self)
            finally:
                self._rollback_registration(
                    extension,
                    dispatch_name,
                    context_sources,
                )
            raise
        return ToolExtensionHandle(self, extension.name)

    def has(self, name: str) -> bool:
        return name in self.extensions

    def get_dispatch(self, name: str) -> ToolDispatch:
        extension_name = self._dispatches.get(name)
        if extension_name is None:
            available = ", ".join(sorted(self._dispatches)) or "none"
            raise ValueError(
                f"Dispatch `{name}` is not registered. Available: {available}."
            )
        extension = self.extensions[extension_name]
        if not isinstance(extension, ToolDispatch):
            raise RuntimeError(f"Extension `{extension_name}` is not a ToolDispatch")
        return extension

    def get_context_provider(self, source: str) -> ToolContextProvider:
        extension_name = self._context_sources.get(source)
        if extension_name is None:
            available = ", ".join(sorted(self._context_sources)) or "none"
            raise ValueError(
                f"Context source `{source}` is not registered. Available: {available}."
            )
        extension = self.extensions[extension_name]
        if not isinstance(extension, ToolContextProvider):
            raise RuntimeError(
                f"Extension `{extension_name}` is not a ToolContextProvider"
            )
        return extension

    async def resolve_context(
        self,
        definition: ToolDefinition,
        intent: ToolIntent,
        context: ToolRuntimeContext,
    ) -> dict[str, Any]:
        resolved = {}
        for binding in definition.context.bindings:
            if not binding.required and binding.source not in self._context_sources:
                continue
            provider = self.get_context_provider(binding.source)
            value = await provider.resolve(
                ContextRequest(
                    binding=binding,
                    definition=definition,
                    intent=intent,
                    context=context,
                )
            )
            if value is not _MISSING_CONTEXT:
                resolved[binding.parameter] = value
        return resolved

    async def dispatch(self, request: DispatchRequest) -> ToolOutcome:
        dispatcher = self.get_dispatch(request.plan.dispatch.name)
        outcome = await dispatcher.dispatch(request)
        if not isinstance(outcome, ToolOutcome):
            raise TypeError("Tool dispatch extensions must return ToolOutcome")
        return outcome

    async def before_tool(
        self,
        payload: BeforeToolPolicy,
    ) -> BeforeToolPolicy | ToolOutcome:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.before_tool(payload)
            if isinstance(result, ToolOutcome):
                self._validate_blocked_outcome(result, payload.intent)
                return result
            if not isinstance(result, BeforeToolPolicy):
                raise TypeError(
                    "before_tool policies must return BeforeToolPolicy or ToolOutcome"
                )
            if (
                result.intent.id != previous.intent.id
                or result.intent.name != previous.intent.name
                or result.definition is not previous.definition
                or result.context is not previous.context
            ):
                raise ValueError(
                    "before_tool policies may only replace intent arguments"
                )
            payload = result
        return payload

    async def before_dispatch(
        self,
        payload: BeforeDispatchPolicy,
    ) -> BeforeDispatchPolicy | ToolOutcome:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.before_dispatch(payload)
            if isinstance(result, ToolOutcome):
                self._validate_blocked_outcome(result, payload.plan.intent)
                return result
            if not isinstance(result, BeforeDispatchPolicy):
                raise TypeError(
                    "before_dispatch policies must return "
                    "BeforeDispatchPolicy or ToolOutcome"
                )
            if (
                result.plan.intent != previous.plan.intent
                or result.plan.definition is not previous.plan.definition
                or result.plan.visible_arguments != previous.plan.visible_arguments
                or result.plan.runtime_arguments != previous.plan.runtime_arguments
                or result.plan.feedback != previous.plan.feedback
                or result.context is not previous.context
            ):
                raise ValueError(
                    "before_dispatch policies may only replace the dispatch spec"
                )
            payload = result
        return payload

    async def after_tool(self, payload: AfterToolPolicy) -> AfterToolPolicy:
        for policy in self._iter_policies():
            previous = payload
            result = await policy.after_tool(payload)
            if not isinstance(result, AfterToolPolicy):
                raise TypeError("after_tool policies must return AfterToolPolicy")
            if (
                result.plan is not previous.plan
                or result.context is not previous.context
                or result.outcome.intent_id != previous.outcome.intent_id
                or result.outcome.tool_name != previous.outcome.tool_name
            ):
                raise ValueError(
                    "after_tool policies may only replace outcome result fields"
                )
            payload = result
        return payload

    def _iter_policies(self) -> tuple[ToolPolicy, ...]:
        policies = tuple(self.extensions[name] for name in self._policies)
        if not all(isinstance(policy, ToolPolicy) for policy in policies):
            raise RuntimeError("The policy index contains a non-ToolPolicy extension")
        return policies

    @staticmethod
    def _validate_blocked_outcome(
        outcome: ToolOutcome,
        intent: ToolIntent,
    ) -> None:
        if outcome.status != "blocked":
            raise ValueError("A blocking policy must return a blocked ToolOutcome")
        if outcome.intent_id != intent.id or outcome.tool_name != intent.name:
            raise ValueError("A policy returned an outcome for another tool intent")

    def remove(self, name: str) -> None:
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        self._remove_indexes(extension)
        try:
            extension.on_remove(self)
        finally:
            del self.extensions[name]
            extension._unbind_registry()

    async def aremove(self, name: str) -> None:
        if name not in self.extensions:
            return
        extension = self.extensions[name]
        self._remove_indexes(extension)
        try:
            await extension.aon_remove(self)
        finally:
            del self.extensions[name]
            extension._unbind_registry()

    def _remove_indexes(self, extension: ToolExtension) -> None:
        if isinstance(extension, ToolDispatch):
            self._dispatches.pop(extension.dispatch_name, None)
        if isinstance(extension, ToolContextProvider):
            for source in extension.sources:
                self._context_sources.pop(source, None)
        if extension.name in self._policies:
            self._policies.remove(extension.name)


_MISSING_CONTEXT = object()


__all__ = [
    "BackgroundDispatch",
    "ContextRequest",
    "DetachedDispatch",
    "ForegroundDispatch",
    "OptionalBackgroundDispatch",
    "RuntimeContextProvider",
    "ToolContextProvider",
    "ToolDispatch",
    "ToolExtension",
    "ToolExtensionHandle",
    "ToolExtensionRegistry",
    "ToolPolicy",
]
