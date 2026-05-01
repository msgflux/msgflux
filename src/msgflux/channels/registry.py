import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar, cast

from msgflux.channels.exceptions import AgentNotFoundError

T = TypeVar("T")
Processor = Callable[..., Any]
DEFAULT_PROCESSOR_KEY = "*"


@dataclass
class ChannelContext:
    channel: str
    agent_name: str
    request_id: str
    request: Any
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRun:
    messages: List[Mapping[str, Any]]
    vars: Mapping[str, Any] = field(default_factory=dict)
    stream: Optional[bool] = None
    model_preference: Optional[str] = None
    tool_filter: Optional[Mapping[str, Any]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelSettings:
    max_request_bytes: Optional[int] = None
    request_timeout_s: Optional[float] = None
    enable_docs: bool = True
    cors: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False
    cors_allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_allowed_headers: List[str] = field(default_factory=lambda: ["*"])
    enable_otel: bool = False
    otel_kwargs: Dict[str, Any] = field(default_factory=dict)


class ChannelRegistry:
    """Registry for channel-exposed agents and request processors."""

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}
        self._pre_processors: Dict[str, List[Processor]] = {}
        self._post_processors: Dict[str, List[Processor]] = {}
        self._settings = ChannelSettings()
        self._auth_handler: Optional[Processor] = None
        self._authorizers: Dict[str, List[Processor]] = {}
        self._error_handlers: List[tuple[type[BaseException], Processor]] = []
        self._startup_hooks: List[Processor] = []
        self._shutdown_hooks: List[Processor] = []
        self._hooks: Dict[str, List[Processor]] = {}

    def settings(self, **updates: Any) -> ChannelSettings:
        if not updates:
            return self._settings

        for key, value in updates.items():
            if not hasattr(self._settings, key):
                raise TypeError(f"Unknown channel setting `{key}`")
            setattr(self._settings, key, value)
        return self._settings

    def _resolve_name(self, obj: Any, name: Optional[str] = None) -> str:
        if name:
            return name

        attr_name = getattr(obj, "name", None)
        if callable(attr_name) and not isinstance(attr_name, str):
            attr_name = attr_name()
        if isinstance(attr_name, str) and attr_name:
            return attr_name

        dunder_name = getattr(obj, "__name__", None)
        if isinstance(dunder_name, str) and dunder_name:
            return dunder_name

        raise TypeError(
            "Unable to resolve a channel name. Provide `name=` or define `.name` "
            "or `.__name__` on the registered object."
        )

    def agent(
        self,
        obj: Optional[T] = None,
        *,
        name: Optional[str] = None,
    ) -> T | Callable[[T], T]:
        if obj is not None:
            self._register_agent(obj, name=name)
            return obj

        def decorator(agent_obj: T) -> T:
            self._register_agent(agent_obj, name=name)
            return agent_obj

        return decorator

    def _register_agent(self, obj: T, *, name: Optional[str] = None) -> Any:
        agent = self._materialize_agent(obj)
        key = self._resolve_agent_name(obj, agent, name)
        self._agents[key] = agent
        return agent

    def _materialize_agent(self, obj: T) -> Any:
        if not inspect.isclass(obj):
            return obj
        try:
            return obj()
        except TypeError as e:
            raise TypeError(
                "ChannelRegistry.agent can register agent classes only when they "
                "are instantiable without arguments. Register an instance when "
                "constructor arguments are required."
            ) from e

    def _resolve_agent_name(
        self,
        original: Any,
        agent: Any,
        name: Optional[str] = None,
    ) -> str:
        try:
            return self._resolve_name(agent, name)
        except TypeError:
            if agent is not original:
                return self._resolve_name(original, name)
            raise

    def pre(
        self,
        agent_name: str | Processor = DEFAULT_PROCESSOR_KEY,
    ) -> Processor | Callable[[Processor], Processor]:
        if callable(agent_name) and not isinstance(agent_name, str):
            processor = cast(Processor, agent_name)
            self._pre_processors.setdefault(DEFAULT_PROCESSOR_KEY, []).append(processor)
            return processor

        key = cast(str, agent_name)

        def decorator(processor: Processor) -> Processor:
            self._pre_processors.setdefault(key, []).append(processor)
            return processor

        return decorator

    def post(
        self,
        agent_name: str | Processor = DEFAULT_PROCESSOR_KEY,
    ) -> Processor | Callable[[Processor], Processor]:
        if callable(agent_name) and not isinstance(agent_name, str):
            processor = cast(Processor, agent_name)
            self._post_processors.setdefault(DEFAULT_PROCESSOR_KEY, []).append(
                processor
            )
            return processor

        key = cast(str, agent_name)

        def decorator(processor: Processor) -> Processor:
            self._post_processors.setdefault(key, []).append(processor)
            return processor

        return decorator

    def auth(
        self,
        handler: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        if handler is not None:
            self._auth_handler = handler
            return handler

        def decorator(fn: Processor) -> Processor:
            self._auth_handler = fn
            return fn

        return decorator

    def authorize(
        self,
        target: str | Processor | None = None,
        *,
        agent: str = DEFAULT_PROCESSOR_KEY,
    ) -> Processor | Callable[[Processor], Processor]:
        if callable(target) and not isinstance(target, str):
            processor = cast(Processor, target)
            self._authorizers.setdefault(DEFAULT_PROCESSOR_KEY, []).append(processor)
            return processor

        key = target if isinstance(target, str) else agent

        def decorator(processor: Processor) -> Processor:
            self._authorizers.setdefault(key, []).append(processor)
            return processor

        return decorator

    def error_handler(
        self,
        target: type[BaseException] | Processor | None = None,
    ) -> Processor | Callable[[Processor], Processor]:
        if callable(target) and not inspect.isclass(target):
            handler = cast(Processor, target)
            self._error_handlers.append((Exception, handler))
            return handler

        exc_type = Exception
        if target is not None:
            if not inspect.isclass(target) or not issubclass(target, BaseException):
                raise TypeError("error_handler expects an exception type or callable")
            exc_type = target

        def decorator(handler: Processor) -> Processor:
            self._error_handlers.append((exc_type, handler))
            return handler

        return decorator

    def startup(
        self,
        hook: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        if hook is not None:
            self._startup_hooks.append(hook)
            return hook

        def decorator(fn: Processor) -> Processor:
            self._startup_hooks.append(fn)
            return fn

        return decorator

    def shutdown(
        self,
        hook: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        if hook is not None:
            self._shutdown_hooks.append(hook)
            return hook

        def decorator(fn: Processor) -> Processor:
            self._shutdown_hooks.append(fn)
            return fn

        return decorator

    def hook(
        self,
        event: str,
        handler: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        if handler is not None:
            self._hooks.setdefault(event, []).append(handler)
            return handler

        def decorator(fn: Processor) -> Processor:
            self._hooks.setdefault(event, []).append(fn)
            return fn

        return decorator

    def on_request_start(
        self,
        handler: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        return self.hook("request_start", handler)

    def on_request_end(
        self,
        handler: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        return self.hook("request_end", handler)

    def on_stream_chunk(
        self,
        handler: Optional[Processor] = None,
    ) -> Processor | Callable[[Processor], Processor]:
        return self.hook("stream_chunk", handler)

    def get_agent(self, name: str) -> Any:
        try:
            return self._agents[name]
        except KeyError as e:
            raise AgentNotFoundError(f"Agent `{name}` is not registered") from e

    def agents(self) -> Dict[str, Any]:
        return dict(self._agents)

    def pre_processors(self, agent_name: str) -> List[Processor]:
        return [
            *self._pre_processors.get(DEFAULT_PROCESSOR_KEY, []),
            *self._pre_processors.get(agent_name, []),
        ]

    def post_processors(self, agent_name: str) -> List[Processor]:
        return [
            *self._post_processors.get(DEFAULT_PROCESSOR_KEY, []),
            *self._post_processors.get(agent_name, []),
        ]

    def auth_handler(self) -> Optional[Processor]:
        return self._auth_handler

    def authorizers(self, agent_name: str) -> List[Processor]:
        return [
            *self._authorizers.get(DEFAULT_PROCESSOR_KEY, []),
            *self._authorizers.get(agent_name, []),
        ]

    def error_handlers(
        self,
        exc: BaseException,
    ) -> List[tuple[type[BaseException], Processor]]:
        return [
            (exc_type, handler)
            for exc_type, handler in self._error_handlers
            if isinstance(exc, exc_type)
        ]

    def has_lifespan_hooks(self) -> bool:
        return bool(self._startup_hooks or self._shutdown_hooks)

    async def run_startup_hooks(self, *args: Any) -> None:
        for hook in self._startup_hooks:
            await call_processor(hook, *args)

    async def run_shutdown_hooks(self, *args: Any) -> None:
        for hook in self._shutdown_hooks:
            await call_processor(hook, *args)

    async def run_hooks(self, event: str, *args: Any) -> None:
        for hook in self._hooks.get(event, []):
            await call_processor(hook, *args)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)


async def call_processor(processor: Processor, *args: Any) -> Any:
    fn = getattr(processor, "acall", None)
    if fn is None:
        fn = processor
    selected_args = _select_supported_args(fn, args)
    result = fn(*selected_args)
    if inspect.isawaitable(result):
        return await result
    return result


def _select_supported_args(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
) -> tuple[Any, ...]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return args

    parameters = list(signature.parameters.values())
    if any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return args

    positional = [
        param
        for param in parameters
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    return args[: len(positional)]


def load_registry_target(target: str) -> ChannelRegistry:
    module_ref, separator, attr_name = target.rpartition(":")
    if not separator:
        module_ref = target
        attr_name = "registry"

    module = _load_module(module_ref)
    registry = getattr(module, attr_name)
    if not isinstance(registry, ChannelRegistry):
        raise TypeError(
            f"`{target}` must point to a msgflux.channels.ChannelRegistry instance"
        )
    return registry


def _load_module(module_ref: str) -> ModuleType:
    if module_ref.endswith(".py") or "/" in module_ref or "\\" in module_ref:
        path = Path(module_ref).expanduser().resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from `{path}`")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)
