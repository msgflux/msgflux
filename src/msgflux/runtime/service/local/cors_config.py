"""Read and edit configured browser origins without constructing agents."""

import tomllib
from pathlib import Path
from typing import Literal

import msgspec

from msgflux.runtime.service.http.config import ServiceHTTPConfig
from msgflux.runtime.service.local.config_editing import (
    edit_service_config,
    toml_parser,
)
from msgflux.runtime.service.local.manifest import _ServiceConfig


class CORSConfigView(msgspec.Struct, frozen=True):
    config_path: str
    cors_allow_origins: tuple[str, ...]


class CORSConfigMutation(CORSConfigView, frozen=True):
    changed: bool
    restart_required: bool


def read_cors_config(path: Path) -> CORSConfigView:
    target = Path(path).expanduser().resolve(strict=True)
    config = msgspec.convert(
        tomllib.loads(target.read_text()), type=_ServiceConfig, strict=True
    )
    return CORSConfigView(str(target), config.http.cors_allow_origins)


def mutate_cors_config(
    path: Path, origin: str, *, operation: Literal["add", "remove"]
) -> CORSConfigMutation:
    if operation not in {"add", "remove"}:
        raise ValueError("CORS operation must be add or remove")
    ServiceHTTPConfig((origin,))
    origins = ()

    def mutate(document):
        nonlocal origins
        selected = document.get("http", {}).get("cors_allow_origins", [])
        origins = tuple(str(value) for value in selected)
        if (operation == "add" and origin in selected) or (
            operation == "remove" and origin not in selected
        ):
            return
        origins = (
            (*origins, origin)
            if operation == "add"
            else tuple(value for value in origins if value != origin)
        )
        ServiceHTTPConfig(origins)
        parser = toml_parser()
        if "http" not in document:
            document["http"] = parser.table()
        if "cors_allow_origins" not in document["http"]:
            document["http"]["cors_allow_origins"] = parser.array()
        selected = document["http"]["cors_allow_origins"]
        if operation == "add":
            selected.append(origin)
        else:
            while origin in selected:
                selected.remove(origin)

    target, changed = edit_service_config(path, mutate)
    return CORSConfigMutation(str(target), origins, changed, changed)
