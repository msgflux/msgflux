import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from msgflux.auto.cache import get_default_cache_dir
from msgflux.auto.config import AutoModuleConfig
from msgflux.auto.exceptions import (
    AutoModuleConfigurationError,
    AutoModuleSecurityError,
)
from msgflux.auto.sources.base import AutoModuleSource
from msgflux.auto.sources.github import GitHubAutoModuleSource
from msgflux.auto.sources.huggingface import HuggingFaceAutoModuleSource
from msgflux.models import Model
from msgflux.utils.msgspec import load

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

        def call(repo_id: str, **kwargs: Any):
            ref_kwargs = {
                key: kwargs.pop(key) for key in list(kwargs) if key in _REF_OPTIONS
            }
            ref = owner(repo_id, **ref_kwargs)
            return getattr(ref, f"_{self.name}")(**kwargs)

        return call


class AutoModule:
    """Reference to a remote msgFlux module."""

    get_class = _AutoModuleAction("get_class")
    create = _AutoModuleAction("create")

    _SOURCE_PATTERNS = (
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
        config = self._ensure_config()
        return {
            "config": config,
            "repo_id": self.repo_id,
            "source": self.source_name,
            "revision": self.revision,
            "path": self.path,
        }

    def _get_class(self, *, trust_remote_code: bool = False) -> type[Any]:
        config = self._ensure_config()
        if config.module_class is None:
            raise AutoModuleConfigurationError(
                self.repo_id,
                "`module.json` does not define `class`; use `create()`.",
            )
        obj = self._load_entrypoint(
            config.module_class,
            trust_remote_code=trust_remote_code,
        )
        if not isinstance(obj, type):
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"`class` entrypoint `{config.module_class}` did not resolve "
                "to a class.",
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
            module = self._call_factory(factory, config=config, **kwargs)
        elif config.module_class is not None:
            module_cls = self._get_class(trust_remote_code=trust_remote_code)
            module = module_cls(**kwargs)
        else:
            raise AutoModuleConfigurationError(
                self.repo_id,
                "`module.json` must define `factory` or `class`.",
            )

        if config.state is not None:
            state_path = self._download(config.state)
            state = load(str(state_path))
            if not hasattr(module, "load_state_dict"):
                raise AutoModuleConfigurationError(
                    self.repo_id,
                    "created object does not support `load_state_dict()`.",
                )
            module.load_state_dict(state)

        self._apply_model_overrides(module, models or {}, config=config)
        return module

    def _ensure_config(self) -> AutoModuleConfig:
        if self._config is not None:
            return self._config
        config_path = self._download("module.json")
        self._module_root = config_path.parent
        self._config = AutoModuleConfig.from_file(config_path, repo_id=self.repo_id)
        for filename in self._config.files:
            self._download(filename)
        return self._config

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
        raise ValueError(
            f"Unknown AutoModule source `{self.source_name}`. "
            "Use `github` or `huggingface`."
        )

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
        file_path = self._download(filename)
        module_name = self._module_name(file_path)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Could not import `{filename}`.",
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        module_root = str(file_path.parent)
        sys.path.insert(0, module_root)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Failed to execute `{filename}`: {exc}",
            ) from exc
        finally:
            try:
                sys.path.remove(module_root)
            except ValueError:
                pass
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise AutoModuleConfigurationError(
                self.repo_id,
                f"Object `{attr}` not found in `{filename}`.",
            ) from exc

    def _call_factory(
        self,
        factory: Callable[..., Any],
        *,
        config: AutoModuleConfig,
        **kwargs: Any,
    ) -> Any:
        module_root = str(self.path)
        sys.path.insert(0, module_root)
        try:
            return factory(config=config, module_path=self.path, **kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return factory(**kwargs)
        finally:
            try:
                sys.path.remove(module_root)
            except ValueError:
                pass

    def _apply_model_overrides(
        self,
        module: Any,
        models: dict[str, Any],
        *,
        config: AutoModuleConfig,
    ) -> None:
        for alias, value in models.items():
            path = config.models.get(alias, alias)
            parent, attr = self._resolve_attr_parent(module, path)
            current = getattr(parent, attr)
            self._set_attr(parent, attr, self._coerce_model(value, current=current))

    @staticmethod
    def _set_attr(parent: Any, attr: str, value: Any) -> None:
        buffers = getattr(parent, "_buffers", None)
        if isinstance(buffers, dict) and attr in buffers:
            buffers[attr] = value
            return
        setattr(parent, attr, value)

    def _coerce_model(self, value: Any, *, current: Any) -> Any:
        if isinstance(value, str):
            model_type = getattr(current, "model_type", "chat_completion")
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
        safe_repo_id = self.repo_id.replace("/", "_").replace("-", "_")
        safe_revision = self.revision.replace("/", "_").replace("-", "_")
        return f"msgflux_auto_{safe_repo_id}_{safe_revision}_{file_path.stem}"

    @classmethod
    def _parse_repo_id(cls, repo_id: str) -> tuple[str, Optional[str]]:
        for prefix, source in cls._SOURCE_PATTERNS:
            if repo_id.startswith(prefix):
                return repo_id[len(prefix) :], source
        return repo_id, None


__all__ = ["AutoModule"]
