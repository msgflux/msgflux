from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Mapping, Protocol, Sequence, TypeVar

from msgflux.vulcano.events import DomainEvent

__all__ = [
    "ExtensionUiApi",
    "UiCompletionProvider",
    "UiComponentFactory",
    "UiCustomFactory",
    "UiDialogOptions",
    "UiDriver",
    "UiManager",
    "UiPlacement",
    "UiRegistration",
    "UiRenderContext",
    "UiRenderer",
    "UiSeverity",
    "UiState",
    "UiStatus",
    "UiShortcutHandler",
    "UiThemeResult",
    "UiWidget",
    "ToolRenderContext",
    "ToolRenderer",
    "ToolRendererOptions",
    "WorkingIndicatorOptions",
]


UiPlacement = Literal["above_editor", "below_editor"]
UiSeverity = Literal["info", "information", "warning", "error"]
UiComponentFactory = Callable[[object, object], object | Awaitable[object]]
UiCompletionProvider = Callable[[str], str | None | Awaitable[str | None]]
UiShortcutHandler = Callable[[], None | Awaitable[None]]
T = TypeVar("T")
UiCustomFactory = Callable[
    [object, object, Callable[[T], None]],
    object | Awaitable[object],
]


@dataclass(frozen=True)
class UiDialogOptions:
    timeout: float | None = None

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("UI dialog timeout must be greater than zero")


@dataclass(frozen=True)
class WorkingIndicatorOptions:
    frames: tuple[str, ...] | None = None
    interval: float = 0.1

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("Working indicator interval must be greater than zero")


@dataclass(frozen=True)
class UiStatus:
    owner: str
    key: str
    text: str


@dataclass(frozen=True)
class UiWidget:
    owner: str
    key: str
    content: object
    placement: UiPlacement


@dataclass(frozen=True)
class UiState:
    statuses: tuple[UiStatus, ...] = ()
    widgets: tuple[UiWidget, ...] = ()
    header: object | None = None
    footer: object | None = None
    editor_component: object | None = None
    title: str | None = None
    working_message: str | None = None
    working_visible: bool = True
    working_indicator: WorkingIndicatorOptions = field(
        default_factory=WorkingIndicatorOptions
    )


@dataclass(frozen=True)
class UiThemeResult:
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class UiRenderContext:
    host: object
    theme: object
    invalidate: Callable[[], None]


UiRenderer = Callable[
    [DomainEvent, UiRenderContext],
    object | None | Awaitable[object | None],
]


@dataclass(frozen=True)
class ToolRenderContext:
    """Mutable per-call rendering state exposed to a tool UI renderer."""

    host: object
    theme: object
    invalidate: Callable[[], None]
    tool_call_id: str
    tool_name: str
    phase: str
    expanded: bool
    state: dict[str, object]


ToolRenderer = Callable[
    [DomainEvent, ToolRenderContext],
    object | None | Awaitable[object | None],
]


@dataclass(frozen=True)
class ToolRendererOptions:
    """Optional renderers for one registered Agent tool."""

    render_call: ToolRenderer | None = None
    render_update: ToolRenderer | None = None
    render_result: ToolRenderer | None = None


class UiDriver(Protocol):
    def apply_state(self, state: UiState) -> None: ...

    async def select(
        self,
        title: str,
        options: Sequence[str],
        dialog: UiDialogOptions,
    ) -> str | None: ...

    async def confirm(
        self,
        title: str,
        message: str,
        dialog: UiDialogOptions,
    ) -> bool: ...

    async def input(
        self,
        title: str,
        placeholder: str,
        dialog: UiDialogOptions,
    ) -> str | None: ...

    async def editor(
        self,
        title: str,
        prefill: str,
        dialog: UiDialogOptions,
    ) -> str | None: ...

    def notify(
        self,
        message: str,
        severity: UiSeverity,
    ) -> None: ...

    def set_editor_text(self, text: str) -> None: ...

    def get_editor_text(self) -> str: ...

    def paste_to_editor(self, text: str) -> None: ...

    async def custom(
        self,
        factory: UiCustomFactory[T],
        *,
        overlay: bool,
        overlay_options: Mapping[str, object] | None,
    ) -> T | None: ...

    @property
    def theme(self) -> object: ...

    def get_themes(self) -> tuple[str, ...]: ...

    def set_theme(self, theme: str) -> UiThemeResult: ...

    def render_context(self) -> UiRenderContext: ...


class UiRegistration:
    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def remove(self) -> None:
        if not self._active:
            return
        self._active = False
        self._remove()

    def __enter__(self) -> UiRegistration:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.remove()


@dataclass(frozen=True)
class _Contribution:
    identifier: int
    owner: str
    value: object


@dataclass(frozen=True)
class _RendererContribution:
    identifier: int
    owner: str
    event_type: str
    renderer: UiRenderer


@dataclass(frozen=True)
class _ToolRendererContribution:
    identifier: int
    owner: str
    tool_name: str
    options: ToolRendererOptions


class _RegistrationTracker(Protocol):
    def __call__(self, registration: UiRegistration) -> None: ...


class UiManager:
    """Runtime-owned UI registry with an optional frontend driver."""

    def __init__(self) -> None:
        self._driver: UiDriver | None = None
        self._mode = "headless"
        self._identifiers = itertools.count()
        self._statuses: dict[tuple[str, str], _Contribution] = {}
        self._widgets: dict[tuple[str, str], _Contribution] = {}
        self._slots: dict[str, dict[str, _Contribution]] = {
            "header": {},
            "footer": {},
            "editor_component": {},
            "title": {},
            "working_message": {},
            "working_visible": {},
            "working_indicator": {},
        }
        self._renderers: dict[str, _RendererContribution] = {}
        self._tool_renderers: dict[str, _ToolRendererContribution] = {}
        self._tool_render_states: dict[str, dict[str, object]] = {}
        self._completion_providers: dict[int, _Contribution] = {}
        self._shortcuts: dict[str, _Contribution] = {}

    @property
    def available(self) -> bool:
        return self._driver is not None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def state(self) -> UiState:
        statuses = tuple(
            UiStatus(item.owner, str(item.value[0]), str(item.value[1]))
            for item in sorted(
                self._statuses.values(),
                key=lambda contribution: contribution.identifier,
            )
        )
        widgets = tuple(
            UiWidget(
                owner=item.owner,
                key=str(item.value[0]),
                content=item.value[1],
                placement=item.value[2],
            )
            for item in sorted(
                self._widgets.values(), key=lambda contribution: contribution.identifier
            )
        )
        return UiState(
            statuses=statuses,
            widgets=widgets,
            header=self._active_slot("header"),
            footer=self._active_slot("footer"),
            editor_component=self._active_slot("editor_component"),
            title=self._active_slot("title"),
            working_message=self._active_slot("working_message"),
            working_visible=self._active_slot("working_visible", default=True),
            working_indicator=self._active_slot(
                "working_indicator",
                default=WorkingIndicatorOptions(),
            ),
        )

    def bind(self, driver: UiDriver, *, mode: str = "tui") -> None:
        if self._driver is not None and self._driver is not driver:
            raise RuntimeError("Vulcano already has a bound UI driver")
        self._driver = driver
        self._mode = mode
        driver.apply_state(self.state)

    def unbind(self, driver: UiDriver) -> None:
        if self._driver is driver:
            self._driver = None
            self._mode = "headless"

    async def select(
        self,
        title: str,
        options: Sequence[str],
        dialog: UiDialogOptions,
    ) -> str | None:
        if self._driver is None:
            return None
        return await self._driver.select(title, options, dialog)

    async def confirm(
        self,
        title: str,
        message: str,
        dialog: UiDialogOptions,
    ) -> bool:
        if self._driver is None:
            return False
        return await self._driver.confirm(title, message, dialog)

    async def input(
        self,
        title: str,
        placeholder: str,
        dialog: UiDialogOptions,
    ) -> str | None:
        if self._driver is None:
            return None
        return await self._driver.input(title, placeholder, dialog)

    async def editor(
        self,
        title: str,
        prefill: str,
        dialog: UiDialogOptions,
    ) -> str | None:
        if self._driver is None:
            return None
        return await self._driver.editor(title, prefill, dialog)

    def notify(
        self,
        message: str,
        severity: UiSeverity,
    ) -> None:
        if self._driver is not None:
            self._driver.notify(message, severity)

    def set_editor_text(self, text: str) -> None:
        if self._driver is not None:
            self._driver.set_editor_text(text)

    def get_editor_text(self) -> str:
        if self._driver is None:
            return ""
        return self._driver.get_editor_text()

    def paste_to_editor(self, text: str) -> None:
        if self._driver is not None:
            self._driver.paste_to_editor(text)

    async def custom(
        self,
        factory: UiCustomFactory[T],
        *,
        overlay: bool,
        overlay_options: Mapping[str, object] | None,
    ) -> T | None:
        if self._driver is None:
            return None
        return await self._driver.custom(
            factory,
            overlay=overlay,
            overlay_options=overlay_options,
        )

    @property
    def theme(self) -> object | None:
        return self._driver.theme if self._driver is not None else None

    def get_themes(self) -> tuple[str, ...]:
        if self._driver is None:
            return ()
        return self._driver.get_themes()

    def set_theme(self, theme: str) -> UiThemeResult:
        if self._driver is None:
            return UiThemeResult(False, "UI is not available")
        return self._driver.set_theme(theme)

    async def complete(self, value: str) -> str | None:
        providers = sorted(
            self._completion_providers.values(),
            key=lambda contribution: contribution.identifier,
            reverse=True,
        )
        for contribution in providers:
            provider = contribution.value
            if not callable(provider):
                continue
            result = provider(value)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                return str(result)
        return None

    async def invoke_shortcut(self, key: str) -> bool:
        contribution = self._shortcuts.get(key.lower())
        if contribution is None or not callable(contribution.value):
            return False
        result = contribution.value()
        if inspect.isawaitable(result):
            await result
        return True

    def set_status(self, owner: str, key: str, text: str) -> UiRegistration:
        if not key:
            raise ValueError("UI status key cannot be empty")
        contribution = _Contribution(
            next(self._identifiers),
            owner,
            (key, text),
        )
        status_key = (owner, key)
        self._statuses[status_key] = contribution
        self._notify_driver()
        return UiRegistration(
            lambda: self._remove_if_current(
                self._statuses,
                status_key,
                contribution,
            )
        )

    def clear_status(self, owner: str, key: str) -> None:
        if self._statuses.pop((owner, key), None) is not None:
            self._notify_driver()

    def set_widget(
        self,
        owner: str,
        key: str,
        content: object,
        placement: UiPlacement,
    ) -> UiRegistration:
        if not key:
            raise ValueError("UI widget key cannot be empty")
        if placement not in {"above_editor", "below_editor"}:
            raise ValueError(f"Unsupported UI widget placement: {placement}")
        if isinstance(content, list):
            content = tuple(content)
        contribution = _Contribution(
            next(self._identifiers),
            owner,
            (key, content, placement),
        )
        widget_key = (owner, key)
        self._widgets[widget_key] = contribution
        self._notify_driver()
        return UiRegistration(
            lambda: self._remove_if_current(
                self._widgets,
                widget_key,
                contribution,
            )
        )

    def clear_widget(self, owner: str, key: str) -> None:
        if self._widgets.pop((owner, key), None) is not None:
            self._notify_driver()

    def set_slot(
        self,
        slot: str,
        owner: str,
        value: object,
    ) -> UiRegistration:
        contributions = self._slots[slot]
        contribution = _Contribution(next(self._identifiers), owner, value)
        contributions[owner] = contribution
        self._notify_driver()
        return UiRegistration(
            lambda: self._remove_if_current(
                contributions,
                owner,
                contribution,
            )
        )

    def get_slot(self, slot: str) -> object | None:
        return self._active_slot(slot)

    def clear_slot(self, slot: str, owner: str) -> None:
        if self._slots[slot].pop(owner, None) is not None:
            self._notify_driver()

    def register_renderer(
        self,
        owner: str,
        event_type: str,
        renderer: UiRenderer,
    ) -> UiRegistration:
        if not event_type:
            raise ValueError("UI renderer event type cannot be empty")
        if event_type in self._renderers:
            existing = self._renderers[event_type]
            raise ValueError(
                f"UI renderer for {event_type!r} is already registered by "
                f"{existing.owner!r}"
            )
        contribution = _RendererContribution(
            next(self._identifiers),
            owner,
            event_type,
            renderer,
        )
        self._renderers[event_type] = contribution
        return UiRegistration(
            lambda: self._remove_if_current(
                self._renderers,
                event_type,
                contribution,
            )
        )

    def register_completion_provider(
        self,
        owner: str,
        provider: UiCompletionProvider,
    ) -> UiRegistration:
        contribution = _Contribution(next(self._identifiers), owner, provider)
        self._completion_providers[contribution.identifier] = contribution
        return UiRegistration(
            lambda: self._remove_if_current(
                self._completion_providers,
                contribution.identifier,
                contribution,
            )
        )

    def register_tool_renderer(
        self,
        owner: str,
        tool_name: str,
        options: ToolRendererOptions,
    ) -> UiRegistration:
        if not tool_name:
            raise ValueError("Tool renderer name cannot be empty")
        if tool_name in self._tool_renderers:
            existing = self._tool_renderers[tool_name]
            raise ValueError(
                f"Tool renderer for {tool_name!r} is already registered by "
                f"{existing.owner!r}"
            )
        contribution = _ToolRendererContribution(
            next(self._identifiers),
            owner,
            tool_name,
            options,
        )
        self._tool_renderers[tool_name] = contribution
        return UiRegistration(
            lambda: self._remove_if_current(
                self._tool_renderers,
                tool_name,
                contribution,
            )
        )

    def register_shortcut(
        self,
        owner: str,
        key: str,
        handler: UiShortcutHandler,
    ) -> UiRegistration:
        normalized = key.strip().lower()
        if not normalized:
            raise ValueError("UI shortcut cannot be empty")
        if normalized in self._shortcuts:
            existing = self._shortcuts[normalized]
            raise ValueError(
                f"UI shortcut {normalized!r} is already registered by "
                f"{existing.owner!r}"
            )
        contribution = _Contribution(next(self._identifiers), owner, handler)
        self._shortcuts[normalized] = contribution
        return UiRegistration(
            lambda: self._remove_if_current(
                self._shortcuts,
                normalized,
                contribution,
            )
        )

    async def render_event(self, event: DomainEvent) -> object | None:
        contribution = self._renderers.get(event.type)
        if contribution is None or self._driver is None:
            return None
        result = contribution.renderer(event, self._driver.render_context())
        if inspect.isawaitable(result):
            result = await result
        return result

    async def render_tool_event(
        self,
        event: DomainEvent,
        *,
        expanded: bool,
    ) -> object | None:
        tool_name = str(event.payload.get("name", ""))
        tool_call_id = str(event.payload.get("tool_call_id", ""))
        contribution = self._tool_renderers.get(tool_name)
        if contribution is None or self._driver is None or not tool_call_id:
            return None

        if event.type.endswith(".started"):
            phase = "call"
            renderer = contribution.options.render_call
        elif event.type.endswith(".updated"):
            phase = "update"
            renderer = contribution.options.render_update
        else:
            phase = "result"
            renderer = contribution.options.render_result
        if renderer is None:
            return None

        state = self._tool_render_states.setdefault(tool_call_id, {})
        base_context = self._driver.render_context()
        context = ToolRenderContext(
            host=base_context.host,
            theme=base_context.theme,
            invalidate=base_context.invalidate,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            phase=phase,
            expanded=expanded,
            state=state,
        )
        result = renderer(event, context)
        if inspect.isawaitable(result):
            result = await result
        return result

    def clear_tool_render_state(self, tool_call_id: str | None = None) -> None:
        if tool_call_id is None:
            self._tool_render_states.clear()
        else:
            self._tool_render_states.pop(tool_call_id, None)

    def _active_slot(self, slot: str, *, default: object = None) -> object:
        contributions = self._slots[slot].values()
        active = max(
            contributions,
            key=lambda contribution: contribution.identifier,
            default=None,
        )
        return active.value if active is not None else default

    def _remove_if_current(
        self,
        values: dict[object, object],
        key: object,
        contribution: object,
    ) -> None:
        if values.get(key) is contribution:
            values.pop(key)
            self._notify_driver()

    def _notify_driver(self) -> None:
        if self._driver is not None:
            self._driver.apply_state(self.state)


class ExtensionUiApi:
    """Owner-aware Pi-shaped facade over Vulcano's active UI driver."""

    def __init__(
        self,
        manager: UiManager,
        *,
        owner: str,
        assert_active: Callable[[], None],
        track: _RegistrationTracker,
    ) -> None:
        self._manager = manager
        self._owner = owner
        self._assert_active = assert_active
        self._track = track

    @property
    def mode(self) -> str:
        self._assert_active()
        return self._manager.mode

    @property
    def available(self) -> bool:
        self._assert_active()
        return self._manager.available

    @property
    def theme(self) -> object | None:
        self._assert_active()
        return self._manager.theme

    async def select(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> str | None:
        self._assert_active()
        return await self._manager.select(
            title,
            options,
            UiDialogOptions(timeout=timeout),
        )

    async def confirm(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        self._assert_active()
        return await self._manager.confirm(
            title,
            message,
            UiDialogOptions(timeout=timeout),
        )

    async def input(
        self,
        title: str,
        placeholder: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        self._assert_active()
        return await self._manager.input(
            title,
            placeholder,
            UiDialogOptions(timeout=timeout),
        )

    async def editor(
        self,
        title: str,
        prefill: str = "",
        *,
        timeout: float | None = None,
    ) -> str | None:
        self._assert_active()
        return await self._manager.editor(
            title,
            prefill,
            UiDialogOptions(timeout=timeout),
        )

    def notify(
        self,
        message: str,
        severity: UiSeverity = "info",
    ) -> None:
        self._assert_active()
        self._manager.notify(message, severity)

    def set_status(self, key: str, text: str | None) -> None:
        self._assert_active()
        if text is None:
            self._manager.clear_status(self._owner, key)
            return
        self._track(self._manager.set_status(self._owner, key, text))

    def set_working_message(self, message: str | None = None) -> None:
        self._set_optional_slot("working_message", message)

    def set_working_visible(
        self,
        visible: bool | None = None,  # noqa: FBT001
    ) -> None:
        self._set_optional_slot("working_visible", visible)

    def set_working_indicator(
        self,
        options: WorkingIndicatorOptions | None = None,
    ) -> None:
        self._assert_active()
        if options is None:
            self._manager.clear_slot("working_indicator", self._owner)
            return
        if not isinstance(options, WorkingIndicatorOptions):
            raise TypeError("options must be WorkingIndicatorOptions or None")
        self._track(self._manager.set_slot("working_indicator", self._owner, options))

    def set_widget(
        self,
        key: str,
        content: object | None,
        *,
        placement: UiPlacement = "above_editor",
    ) -> None:
        self._assert_active()
        if content is None:
            self._manager.clear_widget(self._owner, key)
            return
        self._track(self._manager.set_widget(self._owner, key, content, placement))

    def set_header(self, content: object | None) -> None:
        self._set_optional_slot("header", content)

    def set_footer(self, content: object | None) -> None:
        self._set_optional_slot("footer", content)

    def set_editor_component(self, factory: UiComponentFactory | None) -> None:
        self._set_optional_slot("editor_component", factory)

    def get_editor_component(self) -> object | None:
        self._assert_active()
        return self._manager.get_slot("editor_component")

    def set_title(self, title: str | None) -> None:
        self._set_optional_slot("title", title)

    def set_editor_text(self, text: str) -> None:
        self._assert_active()
        self._manager.set_editor_text(text)

    def get_editor_text(self) -> str:
        self._assert_active()
        return self._manager.get_editor_text()

    def paste_to_editor(self, text: str) -> None:
        self._assert_active()
        self._manager.paste_to_editor(text)

    async def custom(
        self,
        factory: UiCustomFactory[T],
        *,
        overlay: bool = False,
        overlay_options: Mapping[str, object] | None = None,
    ) -> T | None:
        self._assert_active()
        return await self._manager.custom(
            factory,
            overlay=overlay,
            overlay_options=overlay_options,
        )

    def get_themes(self) -> tuple[str, ...]:
        self._assert_active()
        return self._manager.get_themes()

    def set_theme(self, theme: str) -> UiThemeResult:
        self._assert_active()
        return self._manager.set_theme(theme)

    def register_renderer(
        self,
        event_type: str,
        renderer: UiRenderer,
    ) -> UiRegistration:
        self._assert_active()
        registration = self._manager.register_renderer(
            self._owner,
            event_type,
            renderer,
        )
        self._track(registration)
        return registration

    def add_autocomplete_provider(
        self,
        provider: UiCompletionProvider,
    ) -> UiRegistration:
        self._assert_active()
        registration = self._manager.register_completion_provider(
            self._owner,
            provider,
        )
        self._track(registration)
        return registration

    def register_tool_renderer(
        self,
        tool_name: str,
        options: ToolRendererOptions,
    ) -> UiRegistration:
        self._assert_active()
        if not isinstance(options, ToolRendererOptions):
            raise TypeError("options must be ToolRendererOptions")
        registration = self._manager.register_tool_renderer(
            self._owner,
            tool_name,
            options,
        )
        self._track(registration)
        return registration

    def _register_shortcut(
        self,
        key: str,
        handler: UiShortcutHandler,
    ) -> UiRegistration:
        self._assert_active()
        registration = self._manager.register_shortcut(
            self._owner,
            key,
            handler,
        )
        self._track(registration)
        return registration

    def _set_optional_slot(self, slot: str, value: object | None) -> None:
        self._assert_active()
        if value is None:
            self._manager.clear_slot(slot, self._owner)
            return
        self._track(self._manager.set_slot(slot, self._owner, value))
