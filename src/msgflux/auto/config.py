from pathlib import Path
from typing import Any, Mapping, Optional

import msgspec
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from msgflux.auto.exceptions import AutoModuleConfigurationError
from msgflux.utils.msgspec import read_json
from msgflux.version import __version__


class AutoModuleMetadata(msgspec.Struct, forbid_unknown_fields=True):
    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    license: Optional[str] = None
    tags: list[str] = msgspec.field(default_factory=list)


class AutoModuleConfig(msgspec.Struct, forbid_unknown_fields=True):
    schema_version: int
    msgflux_version: str
    factory: Optional[str] = None
    module_class: Optional[str] = msgspec.field(default=None, name="class")
    state: Optional[str] = None
    models: dict[str, str] = msgspec.field(default_factory=dict)
    files: list[str] = msgspec.field(default_factory=list)
    metadata: AutoModuleMetadata = msgspec.field(default_factory=AutoModuleMetadata)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        repo_id: str,
    ) -> "AutoModuleConfig":
        try:
            config = msgspec.convert(data, type=cls)
        except msgspec.ValidationError as exc:
            raise AutoModuleConfigurationError(repo_id, str(exc)) from exc
        config.validate(repo_id)
        return config

    @classmethod
    def from_file(cls, path: Path, *, repo_id: str) -> "AutoModuleConfig":
        try:
            data = read_json(path)
        except FileNotFoundError as exc:
            raise AutoModuleConfigurationError(
                repo_id, "`module.json` not found"
            ) from exc
        except Exception as exc:
            raise AutoModuleConfigurationError(
                repo_id, f"`module.json` is not valid JSON: {exc}"
            ) from exc
        return cls.from_mapping(data, repo_id=repo_id)

    def validate(self, repo_id: str) -> None:
        if self.schema_version != 1:
            raise AutoModuleConfigurationError(
                repo_id,
                f"`schema_version` must be 1, given {self.schema_version}.",
            )
        if self.factory is None and self.module_class is None:
            raise AutoModuleConfigurationError(
                repo_id,
                "`module.json` must define `factory` or `class`.",
            )
        try:
            specifier = SpecifierSet(self.msgflux_version)
        except Exception as exc:
            raise AutoModuleConfigurationError(
                repo_id,
                f"`msgflux_version` is not a valid version specifier: {exc}",
            ) from exc
        if Version(__version__) not in specifier:
            raise AutoModuleConfigurationError(
                repo_id,
                f"`msgflux_version` requires {self.msgflux_version}, "
                f"current version is {__version__}.",
            )
