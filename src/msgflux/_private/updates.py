"""Explicit release discovery and updates for uv tool installations."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx2
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

from msgflux.utils.errors import exception_message
from msgflux.version import __version__


def _write(message: str, *, file=None, flush: bool = False) -> None:
    stream = sys.stdout if file is None else file
    stream.write(message + "\n")
    if flush:
        stream.flush()


_DEFAULT_INDEX = "https://pypi.org/simple/"


def _index_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            "Index URL must be an HTTP(S) Simple API base without query or fragment"
        )
    # Validate the port before any network or service mutation.
    _ = parts.port
    return urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/"))


def _receipt_index(settings) -> str | None:
    indexes = settings.get("index", [])
    if not indexes:
        return None
    if len(indexes) != 1:
        raise RuntimeError("Multiple uv indexes require a manual update")
    index = indexes[0]
    if (
        set(index) - {"url", "default", "explicit", "format", "authenticate"}
        or not index.get("default")
        or index.get("explicit", False)
        or index.get("format", "simple") != "simple"
        or index.get("authenticate", "auto") != "auto"
    ):
        raise RuntimeError("Custom uv index settings require a manual update")
    return _index_url(index["url"])


def selected_index(requested: str | None) -> str:
    if requested is not None:
        return _index_url(requested)
    receipt = Path(sys.prefix) / "uv-receipt.toml"
    if receipt.is_file():
        tool = tomllib.loads(receipt.read_text())["tool"]
        stored = _receipt_index(tool.get("options", {}))
        if stored is not None:
            return stored
    return _DEFAULT_INDEX


class _SimpleHTMLFiles(HTMLParser):
    """Translate legacy Simple links to the same records used by JSON indexes."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.files = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href is None:
            return
        self.files.append(
            {
                "filename": unquote(urlsplit(href).path.rsplit("/", 1)[-1]),
                "requires-python": attributes.get("data-requires-python"),
                "yanked": "data-yanked" in attributes,
            }
        )


def latest_version(*, prereleases: bool, index_url: str = _DEFAULT_INDEX) -> str:
    """Fetch uncached Simple metadata and select a usable release for this Python."""
    with httpx2.Client(timeout=15, follow_redirects=True) as client:
        response = client.get(
            _index_url(index_url) + "msgflux/",
            headers={
                "Accept": (
                    "application/vnd.pypi.simple.v1+json, "
                    "application/vnd.pypi.simple.v1+html;q=0.2, text/html;q=0.1"
                ),
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
            },
        )
        response.raise_for_status()
        if "html" in response.headers.get("content-type", "").lower():
            parser = _SimpleHTMLFiles()
            parser.feed(response.text)
            files = parser.files
        else:
            files = response.json()["files"]
    python = ".".join(map(str, sys.version_info[:3]))
    versions = set()
    for file in files:
        if file.get("yanked", False) is not False:
            continue
        requires = file.get("requires-python")
        if requires and not SpecifierSet(requires).contains(python):
            continue
        filename = file["filename"]
        try:
            if filename.endswith(".whl"):
                name, version, _, _ = parse_wheel_filename(filename)
            else:
                name, version = parse_sdist_filename(filename)
        except (InvalidWheelFilename, InvalidSdistFilename):
            continue
        if name != "msgflux" or version.is_devrelease:
            continue
        if prereleases or not version.is_prerelease:
            versions.add(version)
    if not versions:
        raise RuntimeError(
            "Index has no eligible msgflux release for this Python/channel"
        )
    return str(max(versions))


@dataclass(frozen=True)
class ToolInstallation:
    uv: str
    executable: Path
    extras: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    python: str = ""
    options: tuple[str, ...] = ()

    def command(self, target: str, *, index_url: str = _DEFAULT_INDEX) -> list[str]:
        extras = f"[{','.join(self.extras)}]" if self.extras else ""
        command = [
            self.uv,
            "tool",
            "install",
            "--force",
            "--upgrade",
            f"msgflux{extras}=={Version(target)}",
            "--python",
            self.python or sys._base_executable,
        ]
        for requirement in self.dependencies:
            command.extend(["--with", requirement])
        command.extend(self.options)
        command.extend(["--default-index", _index_url(index_url)])
        return command


def _receipt_settings(tool):
    """Preserve supported uv requirements and reject settings we cannot retain."""
    requirements = tool["requirements"]
    package = next((item for item in requirements if item["name"] == "msgflux"), None)
    if package is None or set(package) - {"name", "extras", "specifier"}:
        raise RuntimeError(
            "Automatic update refuses Git, URL, local or editable sources"
        )
    if set(tool) - {"requirements", "python", "entrypoints", "options"}:
        raise RuntimeError("Unsupported uv receipt settings; update manually")
    if any(item.get("from") != "msgflux" for item in tool.get("entrypoints", [])):
        raise RuntimeError("Additional tool executables require a manual update")
    dependencies = []
    for item in requirements:
        if set(item) - {"name", "extras", "specifier"}:
            raise RuntimeError("Source-based tool dependencies require a manual update")
        if item is not package:
            extras = item.get("extras", [])
            suffix = f"[{','.join(extras)}]" if extras else ""
            dependencies.append(
                str(Requirement(item["name"] + suffix + item.get("specifier", "")))
            )
    # Fail before stopping a service instead of silently dropping uv settings.
    settings = tool.get("options", {})
    if set(settings) - {"find-links", "index"}:
        raise RuntimeError("Custom uv installer settings require a manual update")
    _receipt_index(settings)
    options = tuple(
        value
        for link in settings.get("find-links", [])
        for value in ("--find-links", link)
    )
    return tuple(package.get("extras", [])), tuple(dependencies), options


def tool_installation() -> ToolInstallation:
    """Refuse to modify an unrelated tool, checkout, or editable installation."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Automatic update requires uv and a uv tool installation")
    prefix = Path(sys.prefix).resolve()
    result = subprocess.run(  # noqa: S603 - argv only; trusted uv/entry point
        [uv, "tool", "dir"], check=True, capture_output=True, text=True
    )
    if prefix != Path(result.stdout.strip()).expanduser().resolve() / "msgflux":
        raise RuntimeError(
            "Automatic update supports only uv tool installations; "
            "update this environment with its original installation method"
        )
    receipt = tomllib.loads((prefix / "uv-receipt.toml").read_text())
    extras, dependencies, options = _receipt_settings(receipt["tool"])
    dist = distribution("msgflux")
    if dist.read_text("direct_url.json") is not None:
        raise RuntimeError(
            "Automatic update refuses Git, URL, local or editable sources"
        )
    # A uv tool interpreter can import a checkout through PYTHONPATH. Do not
    # mistake that checkout for the package managed by this receipt.
    if not Path(__file__).resolve().is_relative_to(prefix):
        raise RuntimeError("Automatic update refuses source checkouts")
    executable = prefix / (
        "Scripts/msgflux.exe" if sys.platform == "win32" else "bin/msgflux"
    )
    if not executable.is_file():
        raise RuntimeError("Cannot locate the installed msgflux entry point")
    return ToolInstallation(
        uv,
        executable,
        extras,
        dependencies,
        sys._base_executable,
        options,
    )


async def service_to_restart(runtime_dir: Path | None, *, force: bool):
    from msgflux.runtime.service.http.client import AgentServiceClient  # noqa: PLC0415
    from msgflux.runtime.service.local.discovery import (  # noqa: PLC0415
        inspect_local_service,
    )
    from msgflux.runtime.service.local.files import read_record  # noqa: PLC0415

    status = await inspect_local_service(runtime_dir=runtime_dir)
    if status.status == "unreachable":
        raise RuntimeError("Cannot verify AgentService; stop it before updating")
    if status.status == "stopped":
        return None
    record = read_record(Path(status.runtime_dir))
    if record is None or record.instance_id != status.instance_id:
        raise RuntimeError("AgentService identity changed; retry the update")
    client = AgentServiceClient(record.url, token=record.token)
    try:
        health = await client.health()
        if health.instance_id != record.instance_id:
            raise RuntimeError("AgentService identity changed; retry the update")
        active = await client.active_runs()
    finally:
        await client.aclose()
    if active and not force:
        raise RuntimeError(
            f"AgentService has {len(active)} active generation(s). "
            "Wait for completion or use --force to interrupt them; "
            "automatic resumption is not guaranteed"
        )
    return record


def _restart_command(installation, record, directory, timeout):
    command = [
        str(installation.executable),
        "service",
        "restart",
        "--runtime-dir",
        str(directory),
        "--cwd",
        record.cwd,
        "--restart-timeout",
        str(timeout),
    ]
    if record.config_path is not None:
        command.extend(["--config", record.config_path])
    else:
        command.extend(["--factory", record.factory])
    return command


def update(options) -> None:
    installation = tool_installation()
    installed = Version(__version__)
    index_url = selected_index(options.index_url)
    target = latest_version(
        prereleases=options.pre or installed.is_prerelease, index_url=index_url
    )
    if Version(target) <= installed:
        _write(f"msgflux {installed} is up to date (latest: {target}).")
        return
    # Import lifecycle dependencies before uv replaces files in this environment.
    from msgflux.runtime.service.local.discovery import (  # noqa: PLC0415
        stop_local_service,
    )

    directory = (
        (options.runtime_dir or Path.home() / ".msgflux/runtime")
        .expanduser()
        .absolute()
    )
    record = asyncio.run(service_to_restart(directory, force=options.force))
    restart = (
        _restart_command(installation, record, directory, options.timeout)
        if record is not None
        else None
    )
    _write(f"Updating msgflux {installed} -> {target}.", flush=True)
    if record is not None:
        _write(
            "Stopping AgentService; active generations may be interrupted. "
            "Automatic resumption is not guaranteed.",
            file=sys.stderr,
            flush=True,
        )
        asyncio.run(
            stop_local_service(runtime_dir=directory, stop_timeout=options.timeout)
        )
    try:
        subprocess.run(  # noqa: S603 - argv only; trusted uv/entry point
            installation.command(target, index_url=index_url), check=True
        )
    except (OSError, subprocess.CalledProcessError, KeyboardInterrupt):
        if restart is not None:
            _write(
                "Update failed; attempting to restart AgentService.", file=sys.stderr
            )
            try:
                subprocess.run(restart, check=True)  # noqa: S603
            except (OSError, subprocess.CalledProcessError) as error:
                _write(
                    f"Service restoration failed: {exception_message(error)}",
                    file=sys.stderr,
                )
        raise
    if restart is not None:
        try:
            subprocess.run(restart, check=True)  # noqa: S603
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"msgflux {target} installed, but AgentService restart failed. "
                "Run 'msgflux service restart' with the same runtime/config; "
                f"{exception_message(error)}"
            ) from error
    _write(f"Updated msgflux to {target}.")


def _positive_timeout(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def main(argv, *, command: str) -> None:
    parser = argparse.ArgumentParser(prog=f"msgflux {command}")
    parser.add_argument(
        "--pre",
        action="store_true",
        help="include prereleases (automatic on prerelease installs)",
    )
    parser.add_argument(
        "--index-url",
        metavar="URL",
        help="Simple API base URL; defaults to the uv tool's stored index or PyPI",
    )
    if command == "version":
        parser.add_argument(
            "--check",
            action="store_true",
            help="fetch the latest version directly from the index, without cache",
        )
        parser.add_argument(
            "--json", action="store_true", help="machine-readable output"
        )
    else:
        parser.description = (
            "Update a uv tool installation and restart its running local AgentService."
        )
        parser.epilog = (
            "Restart can interrupt active generations; automatic resumption is "
            "not guaranteed. Stop other runtimes before updating."
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="allow interruption of active generations during service restart",
        )
        parser.add_argument(
            "--runtime-dir", type=Path, help="local AgentService runtime directory"
        )
        parser.add_argument(
            "--timeout",
            type=_positive_timeout,
            default=30,
            help="service stop/restart timeout in seconds (default: 30)",
        )
    options = parser.parse_args(argv)
    if command == "version" and not options.check:
        if options.pre or options.index_url is not None:
            parser.error("--pre and --index-url require --check")
    try:
        if command == "update":
            update(options)
            return
        result = {"installed": __version__}
        if options.check:
            latest = latest_version(
                prereleases=options.pre or Version(__version__).is_prerelease,
                index_url=selected_index(options.index_url),
            )
            result.update(
                latest=latest, update_available=Version(latest) > Version(__version__)
            )
        if options.json:
            _write(json.dumps(result))
        else:
            _write(f"msgflux {__version__}")
            if options.check:
                status = (
                    "update available" if result["update_available"] else "up to date"
                )
                _write(f"Latest: {result['latest']} ({status})")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        httpx2.HTTPError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        _write(f"Error: {exception_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None
