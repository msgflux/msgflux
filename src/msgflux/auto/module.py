import importlib
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Optional

from msgflux.auto.cache import get_default_cache_dir
from msgflux.auto.config import AutoModuleConfig, AutoModuleModelSlot
from msgflux.auto.exceptions import (
    AutoModuleConfigurationError,
    AutoModuleDownloadError,
    AutoModuleSecurityError,
)
from msgflux.auto.sources.base import AutoModuleSource
from msgflux.auto.sources.github import GitHubAutoModuleSource
from msgflux.auto.sources.huggingface import HuggingFaceAutoModuleSource
from msgflux.auto.sources.local import LocalAutoModuleSource
from msgflux.models import Model
from msgflux.utils.msgspec import load, save
from msgflux.version import __version__

_REF_OPTIONS = {
    "revision",
    "source",
    "cache_dir",
    "local_files_only",
    "force_download",
}


class _AutoModuleAction:
    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, instance: Optional["AutoModule"], owner: type["AutoModule"]):
        if instance is not None:
            return getattr(instance, f"_{self.name}")

        def call(repo_id: str, *args: Any, **kwargs: Any):
            ref_kwargs = {
                key: kwargs.pop(key) for key in list(kwargs) if key in _REF_OPTIONS
            }
            ref = owner(repo_id, **ref_kwargs)
            return getattr(ref, f"_{self.name}")(*args, **kwargs)

        return call


class AutoModule:
    """Reference to a remote msgFlux module."""

    get_class = _AutoModuleAction("get_class")
    create = _AutoModuleAction("create")
    load_into = _AutoModuleAction("load_into")

    _SOURCE_PATTERNS = (
        ("local://", "local"),
        ("hf://", "huggingface"),
        ("huggingface.co/", "huggingface"),
        ("gh://", "github"),
        ("github.com/", "github"),
    )

    def __init__(
        self,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        source: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        self.repo_id, detected_source = self._parse_repo_id(repo_id)
        self.source_name = source or detected_source or "github"
        self.revision = revision or "main"
        self.cache_dir = (
            Path(cache_dir).expanduser() if cache_dir else get_default_cache_dir()
        )
        self.local_files_only = local_files_only
        self.force_download = force_download
        self._config: Optional[AutoModuleConfig] = None
        self._module_root: Optional[Path] = None

    @property
    def config(self) -> AutoModuleConfig:
        return self._ensure_config()

    @property
    def path(self) -> Path:
        self._ensure_config()
        if self._module_root is None:
            raise AutoModuleConfigurationError(self.repo_id, "module path is unknown.")
        return self._module_root

    def check_requirements(self) -> dict[str, Any]:
        config = self._load_config(download_declared_files=False)
        return {
            "config": config,
            "repo_id": self.repo_id,
            "source": self.source_name,
            "revision": self.revision,
            "path": self._module_root,
        }

    @classmethod
    def export(cls, module: Any, path: str | Path) -> Path:
        if not hasattr(module, "state_dict"):
            raise TypeError("AutoModule.export() requires an object with state_dict().")

        output_dir = Path(path).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        state = module.state_dict()
        save(state, str(output_dir / "state.json"))

        config = {
            "schema_version": 1,
            "msgflux_version": f">={__version__}",
            "entrypoint": "module.py",
            "state": "state.json",
            "models": cls._model_slots_from_state(state),
            "files": [],
            "metadata": {},
        }
        save(config, str(output_dir / "module.json"))

        module_py = output_dir / "module.py"
        if not module_py.exists():
            module_py.write_text("", encoding="utf-8")

        return output_dir

    def _get_class(
        self,
        name: Optional[str] = None,
        *,
        trust_remote_code: bool = False,
    ) -> type[Any]:
        config = self._ensure_config()
        if config.module_class is not None:
            obj = self._load_entrypoint(
                config.module_class,
                trust_remote_code=trust_remote_code,
            )
        else:
            hook = self._load_standard_hook(
                "get_class",
                trust_remote_code=trust_remote_code,
                required=False,
            )
            if hook is None:
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    "`module.json` does not define `class` and "
                    "`module.py` does not define `get_class()`.",
                )
            obj = hook() if name is None else hook(name)
        if not isinstance(obj, type):
            raise AutoModuleConfigurationError(
                self.repo_id,
                "`get_class()` did not resolve to a class.",
            )
        return obj

    def _create(
        self,
        *,
        models: Optional[dict[str, Any]] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> Any:
        config = self._ensure_config()
        if config.factory is not None:
            factory = self._load_entrypoint(
                config.factory,
                trust_remote_code=trust_remote_code,
            )
            if not callable(factory):
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    f"`factory` entrypoint `{config.factory}` is not callable.",
                )
            module = self._call_factory(
                factory,
                config=config,
                models=models or {},
                **kwargs,
            )
        elif config.module_class is not None:
            module_cls = self._get_class(trust_remote_code=trust_remote_code)
            module = module_cls(**kwargs)
        else:
            factory = self._load_standard_hook(
                "create",
                trust_remote_code=trust_remote_code,
                required=False,
            )
            if factory is not None:
                module = self._call_factory(
                    factory,
                    config=config,
                    models=models or {},
                    **kwargs,
                )
            else:
                try:
                    module_cls = self._get_class(trust_remote_code=trust_remote_code)
                except (AutoModuleConfigurationError, AutoModuleDownloadError) as exc:
                    raise AutoModuleConfigurationError(
                        self.repo_id,
                        "`create()` requires `module.py:create`, "
                        "`module.py:get_class`, `factory`, or `class`.",
                    ) from exc
                module = module_cls(**kwargs)

        return self._load_into(module, models=models)

    def _load_into(
        self,
        module: Any,
        *,
        models: Optional[dict[str, Any]] = None,
    ) -> Any:
        config = self._ensure_config()
        if config.state is None:
            self._apply_model_overrides(module, models or {}, config=config)
            return module
        if not hasattr(module, "load_state_dict"):
            raise AutoModuleConfigurationError(
                self.repo_id,
                "target object does not support `load_state_dict()`.",
            )

        state_path = self._download(config.state)
        state = load(str(state_path))
        replacements = self._model_replacements(
            config=config,
            state=state,
            models=models or {},
        )
        for path in replacements:
            self._resolve_attr_parent(module, path)
        module.load_state_dict(state, replacements=replacements)
        return module

    def _ensure_config(self) -> AutoModuleConfig:
        return self._load_config(download_declared_files=True)

    def _load_config(self, *, download_declared_files: bool) -> AutoModuleConfig:
        if self._config is not None and not self.force_download:
            if download_declared_files:
                self._download_declared_files(self._config)
            return self._config
        config_path = self._download("module.json")
        self._module_root = config_path.parent
        self._config = AutoModuleConfig.from_file(config_path, repo_id=self.repo_id)
        if download_declared_files:
            self._download_declared_files(self._config)
        return self._config

    def _download_declared_files(self, config: AutoModuleConfig) -> None:
        for filename in config.files:
            self._download(filename)

    def _download(self, filename: str) -> Path:
        source = self._create_source()
        return source.download_file(filename, force_download=self.force_download)

    def _create_source(self) -> AutoModuleSource:
        source_kwargs = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.source_name == "github":
            return GitHubAutoModuleSource(**source_kwargs)
        if self.source_name == "huggingface":
            return HuggingFaceAutoModuleSource(**source_kwargs)
        if self.source_name == "local":
            return LocalAutoModuleSource(**source_kwargs)
        raise ValueError(
            f"Unknown AutoModule source `{self.source_name}`. "
            "Use `github`, `huggingface`, or `local`."
        )

    def _load_standard_hook(
        self,
        name: str,
        *,
        trust_remote_code: bool,
        required: bool,
    ) -> Optional[Callable[..., Any]]:
        if not trust_remote_code:
            raise AutoModuleSecurityError(self.repo_id)
        config = self._ensure_config()
        try:
            module = self._load_remote_python_module(config.entrypoint)
        except AutoModuleDownloadError:
            if required:
                raise
            return None
        hook = getattr(module, name, None)
        if hook is None:
            if required:
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    f"`{config.entrypoint}` does not define `{name}()`.",
                )
            return None
        if not callable(hook):
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"`{config.entrypoint}:{name}` is not callable.",
            )
        return hook

    def _load_entrypoint(self, entrypoint: str, *, trust_remote_code: bool) -> Any:
        module_ref, _, attr = entrypoint.partition(":")
        if not module_ref or not attr:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Invalid entrypoint `{entrypoint}`. Expected `module.py:object`.",
            )
        if module_ref.endswith(".py"):
            if not trust_remote_code:
                raise AutoModuleSecurityError(self.repo_id)
            return self._load_remote_python_object(module_ref, attr)
        module = importlib.import_module(module_ref)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Object `{attr}` not found in `{module_ref}`.",
            ) from exc

    def _load_remote_python_object(self, filename: str, attr: str) -> Any:
        module = self._load_remote_python_module(filename)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Object `{attr}` not found in `{filename}`.",
            ) from exc

    def _load_remote_python_module(self, filename: str) -> ModuleType:
        file_path = self._download(filename)
        module_name = self._module_name(file_path)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Could not import `{filename}`.",
            )
        package_name = module_name.rsplit(".", 1)[0]
        self._clear_remote_submodules(package_name)
        self._ensure_remote_package(package_name, file_path.parent)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with self._temporary_module_root(file_path.parent):
                spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Failed to execute `{filename}`: {exc}",
            ) from exc
        return module

    def _ensure_remote_package(self, package_name: str, module_root: Path) -> None:
        if package_name in sys.modules:
            package = sys.modules[package_name]
            package.__path__ = [str(module_root)]  # type: ignore[attr-defined]
            return
        package = ModuleType(package_name)
        package.__path__ = [str(module_root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    @staticmethod
    def _clear_remote_submodules(package_name: str) -> None:
        prefix = f"{package_name}."
        for name in list(sys.modules):
            if name.startswith(prefix):
                sys.modules.pop(name, None)

    @contextmanager
    def _temporary_module_root(self, module_root: Path) -> Iterator[None]:
        root = str(module_root)
        before = set(sys.modules)
        sys.path.insert(0, root)
        try:
            yield
        finally:
            try:
                sys.path.remove(root)
            except ValueError:
                pass
            root_path = module_root.resolve()
            for name in set(sys.modules) - before:
                module = sys.modules.get(name)
                module_file = getattr(module, "__file__", None)
                if module_file is None or name.startswith("msgflux_auto_"):
                    continue
                try:
                    Path(module_file).resolve().relative_to(root_path)
                except ValueError:
                    continue
                sys.modules.pop(name, None)

    def _call_factory(
        self,
        factory: Callable[..., Any],
        *,
        config: AutoModuleConfig,
        models: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        with self._temporary_module_root(self.path):
            call_kwargs = self._factory_kwargs(
                factory,
                config=config,
                models=models,
                **kwargs,
            )
            return factory(**call_kwargs)

    def _factory_kwargs(
        self,
        factory: Callable[..., Any],
        *,
        config: AutoModuleConfig,
        models: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return {"config": config, "module_path": self.path, **kwargs}
        parameters = signature.parameters
        has_var_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        injected = {"config": config, "module_path": self.path}
        if "models" in parameters:
            injected["models"] = self._coerce_factory_models(config, models)
        if has_var_kwargs:
            return {**injected, **kwargs}
        return {
            **{name: value for name, value in injected.items() if name in parameters},
            **kwargs,
        }

    def _apply_model_overrides(
        self,
        module: Any,
        models: dict[str, Any],
        *,
        config: AutoModuleConfig,
    ) -> None:
        for alias, value in models.items():
            slot = self._model_slot(alias, config=config)
            path = slot.path
            parent, attr = self._resolve_attr_parent(module, path)
            current = getattr(parent, attr)
            model_type = slot.model_type or getattr(current, "model_type", None)
            self._set_attr(
                parent,
                attr,
                self._coerce_model(value, model_type=model_type),
            )

    def _model_replacements(
        self,
        *,
        config: AutoModuleConfig,
        state: dict[str, Any],
        models: dict[str, Any],
    ) -> dict[str, Any]:
        replacements = {}
        for alias, value in models.items():
            slot = self._model_slot(alias, config=config)
            serialized = state.get(slot.path)
            if not self._is_serialized_model(serialized):
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    f"Model slot `{alias}` points to `{slot.path}`, but that "
                    "state key is not a serialized model.",
                )
            model_type = slot.model_type or serialized.get("model_type")
            replacements[slot.path] = self._coerce_model(value, model_type=model_type)
        return replacements

    def _coerce_factory_models(
        self,
        config: AutoModuleConfig,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            alias: self._coerce_model(
                value,
                model_type=self._model_slot(alias, config=config).model_type,
            )
            for alias, value in models.items()
        }

    def _model_slot(
        self,
        alias: str,
        *,
        config: AutoModuleConfig,
    ) -> AutoModuleModelSlot:
        try:
            return config.models[alias]
        except KeyError as exc:
            valid = ", ".join(sorted(config.models)) or "<none>"
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Unknown model slot `{alias}`. Available slots: {valid}.",
            ) from exc

    @staticmethod
    def _set_attr(parent: Any, attr: str, value: Any) -> None:
        buffers = getattr(parent, "_buffers", None)
        if isinstance(buffers, dict) and attr in buffers:
            buffers[attr] = value
            return
        setattr(parent, attr, value)

    def _coerce_model(self, value: Any, *, model_type: Optional[str]) -> Any:
        if isinstance(value, str):
            model_type = model_type or "chat_completion"
            factory = getattr(Model, model_type, None)
            if factory is None:
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    "Cannot create model override for unsupported type "
                    f"`{model_type}`.",
                )
            return factory(value)
        if isinstance(value, dict) and value.get("msgflux_type") == "model":
            return Model.from_serialized(
                provider=value["provider"],
                model_type=value["model_type"],
                state=value["state"],
            )
        return value

    @staticmethod
    def _is_serialized_model(value: Any) -> bool:
        return isinstance(value, dict) and value.get("msgflux_type") == "model"

    def _resolve_attr_parent(self, module: Any, path: str) -> tuple[Any, str]:
        parts = path.split(".")
        if not parts or any(not part for part in parts):
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Invalid model target path `{path}`.",
            )
        parent = module
        for part in parts[:-1]:
            try:
                parent = getattr(parent, part)
            except AttributeError as exc:
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    f"Model target path `{path}` does not exist.",
                ) from exc
        if not hasattr(parent, parts[-1]):
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Model target path `{path}` does not exist.",
            )
        return parent, parts[-1]

    def _module_name(self, file_path: Path) -> str:
        safe_repo_id = self._python_identifier(self.repo_id)
        safe_revision = self._python_identifier(self.revision)
        safe_stem = self._python_identifier(file_path.stem)
        return f"msgflux_auto_{safe_repo_id}_{safe_revision}.{safe_stem}"

    @staticmethod
    def _python_identifier(value: str) -> str:
        chars = [char if char.isalnum() else "_" for char in value]
        identifier = "".join(chars).strip("_") or "module"
        if identifier[0].isdigit():
            identifier = f"_{identifier}"
        return identifier

    @classmethod
    def _model_slots_from_state(
        cls,
        state: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        slots: dict[str, dict[str, Any]] = {}
        for path, value in state.items():
            if not cls._is_serialized_model(value):
                continue
            alias = cls._unique_model_alias(path, slots)
            model_state = value.get("state") or {}
            slot = {
                "path": path,
                "provider": value.get("provider"),
                "model_type": value.get("model_type"),
                "model_id": model_state.get("model_id"),
                "state": model_state,
            }
            slots[alias] = {key: val for key, val in slot.items() if val is not None}
        return slots

    @staticmethod
    def _unique_model_alias(path: str, slots: dict[str, dict[str, Any]]) -> str:
        base = path.replace(".", "-")
        alias = base
        index = 2
        while alias in slots:
            alias = f"{base}-{index}"
            index += 1
        return alias

    @classmethod
    def _parse_repo_id(cls, repo_id: str) -> tuple[str, Optional[str]]:
        for prefix, source in cls._SOURCE_PATTERNS:
            if repo_id.startswith(prefix):
                return repo_id[len(prefix) :], source
        return repo_id, None


__all__ = ["AutoModule"]
