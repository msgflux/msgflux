"""Release discovery never caches; updates coordinate package and daemon lifecycle."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest

from msgflux import cli
from msgflux._private import updates
from msgflux.runtime.service.http import client as http_client
from msgflux.runtime.service.local import discovery, files
from msgflux.runtime.service.local.records import LocalServiceRecord, LocalServiceStatus


def release(version, **kwargs):
    return {"filename": f"msgflux-{version}-py3-none-any.whl", **kwargs}


@pytest.fixture
def index(monkeypatch):
    payload = {"files": []}
    calls = []

    def respond(request):
        calls.append(request)
        return httpx2.Response(200, json=payload)

    original = httpx2.Client
    monkeypatch.setattr(
        updates.httpx2,
        "Client",
        lambda **kwargs: original(transport=httpx2.MockTransport(respond), **kwargs),
    )
    return payload, calls


def test_release_filtering_and_fresh_queries(index):
    payload, calls = index
    payload["files"] = [
        release("1.0.0rc4"),
        release("1.0.0"),
        release("1.1.0rc1"),
        release("1.2.0", yanked="broken"),
        release("1.2.1", yanked=""),
        release("1.3.0", **{"requires-python": ">=99"}),
        release("2.0.0.dev1"),
        {"filename": "invalid"},
    ]
    assert updates.latest_version(prereleases=False) == "1.0.0"
    assert updates.latest_version(prereleases=True) == "1.1.0rc1"
    assert len(calls) == 2
    for request in calls:
        assert str(request.url) == "https://pypi.org/simple/msgflux/"
        assert request.headers["Cache-Control"] == "no-cache, no-store"
        assert request.headers["Accept"].startswith(
            "application/vnd.pypi.simple.v1+json"
        )


def test_empty_or_incompatible_index_errors(index):
    with pytest.raises(RuntimeError, match="no eligible"):
        updates.latest_version(prereleases=False)


def test_sdist_and_yanked_files(index):
    index[0]["files"] = [
        {"filename": "msgflux-1.0.1.tar.gz"},
        release("1.0.1", yanked=True),
        release("1.0.2", yanked=True),
        {"filename": "other-9.0.0.tar.gz"},
    ]
    assert updates.latest_version(prereleases=False) == "1.0.1"


def test_local_version_never_queries_network(monkeypatch, capsys):
    remote = Mock(side_effect=AssertionError("network"))
    monkeypatch.setattr(updates, "latest_version", remote)
    cli.main(["version", "--json"])
    assert json.loads(capsys.readouterr().out) == {"installed": updates.__version__}
    with pytest.raises(SystemExit) as exited:
        cli.main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"msgflux {updates.__version__}\n"
    remote.assert_not_called()


@pytest.mark.parametrize(
    "installed,pre,expected",
    [("1.0.0", False, False), ("1.0.0rc4", False, True), ("1.0.0", True, True)],
)
def test_version_check_channel(monkeypatch, capsys, installed, pre, expected):
    monkeypatch.setattr(updates, "__version__", installed)
    remote = Mock(return_value="1.1.0")
    monkeypatch.setattr(updates, "latest_version", remote)
    cli.main(["version", "--check", "--json"] + (["--pre"] if pre else []))
    remote.assert_called_once_with(
        prereleases=expected, index_url="https://pypi.org/simple/"
    )
    assert json.loads(capsys.readouterr().out)["update_available"] is True


def test_network_failure_is_not_current_version(monkeypatch, capsys):
    monkeypatch.setattr(
        updates, "latest_version", Mock(side_effect=httpx2.ReadTimeout("offline"))
    )
    with pytest.raises(SystemExit) as exited:
        cli.main(["version", "--check", "--json"])
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "offline" in captured.err


@pytest.mark.parametrize(
    "args",
    [
        ["version", "--pre"],
        ["update", "--timeout", "nan"],
        ["update", "--timeout", "0"],
        ["update", "--timeout", "inf"],
    ],
)
def test_invalid_options(args):
    with pytest.raises(SystemExit) as exited:
        cli.main(args)
    assert exited.value.code == 2


@pytest.fixture
def installation(monkeypatch, tmp_path):
    prefix = tmp_path / "tools/msgflux"
    executable = prefix / "bin/msgflux"
    executable.parent.mkdir(parents=True)
    executable.touch()
    receipt = prefix / "uv-receipt.toml"
    receipt.write_text(
        '[tool]\nrequirements = [{name="msgflux", extras=["coding","service"], specifier="==1.0.0rc4"}]'
    )
    monkeypatch.setattr(updates.sys, "prefix", str(prefix))
    monkeypatch.setattr(
        updates, "__file__", str(prefix / "lib/msgflux/_private/updates.py")
    )
    monkeypatch.setattr(updates.shutil, "which", lambda _: "/bin/uv")
    monkeypatch.setattr(
        updates.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(stdout=str(prefix.parent))),
    )
    monkeypatch.setattr(
        updates, "distribution", lambda _: SimpleNamespace(read_text=lambda _: None)
    )
    return prefix, receipt, executable


def test_uv_detection_preserves_executable(installation):
    tool = updates.tool_installation()
    assert tool.uv == "/bin/uv" and tool.executable == installation[2]
    assert tool.extras == ("coding", "service")


@pytest.mark.parametrize(
    "source",
    [
        'git="https://example.com/repo"',
        "editable=true",
        'path="/repo"',
        'url="https://example.com/a.whl"',
    ],
)
def test_source_installation_refused(installation, source):
    installation[1].write_text(f'[tool]\nrequirements = [{{name="msgflux", {source}}}]')
    with pytest.raises(RuntimeError, match="refuses"):
        updates.tool_installation()


def test_direct_url_refused(installation, monkeypatch):
    monkeypatch.setattr(
        updates, "distribution", lambda _: SimpleNamespace(read_text=lambda _: "{}")
    )
    with pytest.raises(RuntimeError, match="refuses"):
        updates.tool_installation()


def test_project_environment_refused(installation, monkeypatch):
    monkeypatch.setattr(updates.sys, "prefix", "/project/.venv")
    with pytest.raises(RuntimeError, match="only uv tool"):
        updates.tool_installation()


def test_checkout_import_refused(installation, monkeypatch):
    monkeypatch.setattr(updates, "__file__", "/repo/src/msgflux/_private/updates.py")
    with pytest.raises(RuntimeError, match="source checkouts"):
        updates.tool_installation()


@pytest.fixture
def daemon(monkeypatch, tmp_path):
    record = LocalServiceRecord(
        "instance",
        str(tmp_path),
        "http://127.0.0.1:12345",
        123,
        "secret",
        factory="fixture:create",
    )
    monkeypatch.setattr(
        discovery,
        "inspect_local_service",
        AsyncMock(
            return_value=LocalServiceStatus(
                "running", str(tmp_path), instance_id="instance"
            )
        ),
    )
    monkeypatch.setattr(files, "read_record", lambda _: record)
    client = SimpleNamespace(
        health=AsyncMock(return_value=SimpleNamespace(instance_id="instance")),
        active_runs=AsyncMock(return_value=[object()]),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        http_client, "AgentServiceClient", lambda *_args, **_kwargs: client
    )
    return record, client


@pytest.mark.asyncio
async def test_active_generation_requires_force(daemon, tmp_path):
    with pytest.raises(RuntimeError, match="active generation"):
        await updates.service_to_restart(tmp_path, force=False)
    assert await updates.service_to_restart(tmp_path, force=True) == daemon[0]
    assert daemon[1].aclose.await_count == 2


@pytest.mark.asyncio
async def test_empty_service_is_allowed(daemon, tmp_path):
    daemon[1].active_runs.return_value = []
    assert await updates.service_to_restart(tmp_path, force=False) == daemon[0]


@pytest.mark.asyncio
async def test_unreachable_service_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        discovery,
        "inspect_local_service",
        AsyncMock(return_value=LocalServiceStatus("unreachable", str(tmp_path))),
    )
    with pytest.raises(RuntimeError, match="Cannot verify"):
        await updates.service_to_restart(tmp_path, force=True)


@pytest.mark.asyncio
async def test_changed_service_identity_refused(daemon, tmp_path):
    daemon[1].health.return_value.instance_id = "other"
    with pytest.raises(RuntimeError, match="identity changed"):
        await updates.service_to_restart(tmp_path, force=True)
    daemon[1].aclose.assert_awaited_once()


@pytest.fixture
def lifecycle(monkeypatch, tmp_path):
    events = []
    tool = updates.ToolInstallation("/bin/uv", Path("/tools/msgflux/bin/msgflux"))
    record = SimpleNamespace(
        cwd=str(tmp_path), config_path="/config.toml", factory=None
    )
    monkeypatch.setattr(updates, "tool_installation", lambda: tool)
    monkeypatch.setattr(updates, "latest_version", lambda **_: "1.0.1")
    monkeypatch.setattr(updates, "__version__", "1.0.0")

    async def inspect(*args, **kwargs):
        events.append("inspect")
        return record

    async def stop(**kwargs):
        events.append("stop")

    def run(command, **kwargs):
        events.append(command)

    monkeypatch.setattr(updates, "service_to_restart", inspect)
    monkeypatch.setattr(discovery, "stop_local_service", stop)
    monkeypatch.setattr(updates.subprocess, "run", run)
    options = SimpleNamespace(
        pre=False, force=False, runtime_dir=tmp_path, timeout=30, index_url=None
    )
    return events, options


def test_update_stop_install_restart_order(lifecycle):
    events, options = lifecycle
    updates.update(options)
    assert events[:3] == [
        "inspect",
        "stop",
        updates.ToolInstallation("/bin/uv", Path("/tools/msgflux/bin/msgflux")).command(
            "1.0.1"
        ),
    ]
    restart = events[3]
    assert restart[:3] == ["/tools/msgflux/bin/msgflux", "service", "restart"]
    assert restart[-2:] == ["--config", "/config.toml"]
    assert str(options.runtime_dir) in restart


def test_no_update_does_not_touch_service(lifecycle, monkeypatch):
    monkeypatch.setattr(updates, "latest_version", lambda **_: "1.0.0")
    updates.update(lifecycle[1])
    assert lifecycle[0] == []


def test_stopped_service_stays_stopped(lifecycle, monkeypatch):
    monkeypatch.setattr(updates, "service_to_restart", AsyncMock(return_value=None))
    updates.update(lifecycle[1])
    assert lifecycle[0] == [
        updates.ToolInstallation("/bin/uv", Path("/tools/msgflux/bin/msgflux")).command(
            "1.0.1"
        )
    ]


def test_refusal_leaves_installation_untouched(lifecycle, monkeypatch):
    monkeypatch.setattr(
        updates,
        "service_to_restart",
        AsyncMock(side_effect=RuntimeError("active generation")),
    )
    with pytest.raises(RuntimeError, match="active generation"):
        updates.update(lifecycle[1])
    assert lifecycle[0] == []


def test_failed_stop_never_runs_installer(lifecycle, monkeypatch):
    monkeypatch.setattr(
        discovery,
        "stop_local_service",
        AsyncMock(side_effect=TimeoutError("still running")),
    )
    with pytest.raises(TimeoutError):
        updates.update(lifecycle[1])
    assert lifecycle[0] == ["inspect"]


def test_install_failure_attempts_service_restoration(lifecycle, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "/bin/uv":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(updates.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        updates.update(lifecycle[1])
    assert len(calls) == 2 and calls[1][1:3] == ["service", "restart"]


def test_restart_failure_reports_installed_package(lifecycle, monkeypatch):
    def run(command, **kwargs):
        if command[0] != "/bin/uv":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(updates.subprocess, "run", run)
    with pytest.raises(
        RuntimeError, match=r"1\.0\.1 installed, but AgentService restart failed"
    ):
        updates.update(lifecycle[1])


@pytest.mark.parametrize(
    "setting", ['[tool.options]\nresolution="lowest"', "overrides = []"]
)
def test_unknown_receipt_settings_refused(installation, setting):
    receipt = installation[1]
    receipt.write_text(receipt.read_text() + "\n" + setting)
    with pytest.raises(RuntimeError, match="manual"):
        updates.tool_installation()


def test_installer_arguments_preserve_dependencies_and_python():
    installation = updates.ToolInstallation(
        "/bin/uv",
        Path("/tool/bin/msgflux"),
        ("coding", "service"),
        ("plugin>=2",),
        "/python3.13",
        ("--find-links", "/wheels"),
    )
    assert installation.command("1.0.1") == [
        "/bin/uv",
        "tool",
        "install",
        "--force",
        "--upgrade",
        "msgflux[coding,service]==1.0.1",
        "--python",
        "/python3.13",
        "--with",
        "plugin>=2",
        "--find-links",
        "/wheels",
        "--default-index",
        "https://pypi.org/simple/",
    ]


def test_factory_restart_preserves_launch_source():
    record = SimpleNamespace(factory="package:create", config_path=None, cwd="/project")
    command = updates._restart_command(
        updates.ToolInstallation("uv", Path("/bin/msgflux")),
        record,
        Path("/runtime"),
        30,
    )
    assert command[-2:] == ["--factory", "package:create"]
    assert command[command.index("--cwd") + 1] == "/project"


def test_install_and_restore_failure_reports_both(lifecycle, monkeypatch, capsys):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(updates.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        updates.update(lifecycle[1])
    assert "Service restoration failed" in capsys.readouterr().err


def test_custom_json_index_preserves_nested_base(index):
    index[0]["files"] = [release("1.0.1")]
    assert (
        updates.latest_version(
            prereleases=False, index_url="https://packages.example/team/simple"
        )
        == "1.0.1"
    )
    assert str(index[1][0].url) == "https://packages.example/team/simple/msgflux/"


def test_html_simple_index_filters_yanked_and_incompatible(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="""
            <!DOCTYPE html><html><body>
            <a href="../../files/msgflux-1.0.1-py3-none-any.whl#sha256=abc"
               data-requires-python="&gt;=3.11">msgflux-1.0.1-py3-none-any.whl</a>
            <a href="/files/msgflux-1.0.2.tar.gz" data-yanked="">yanked</a>
            <a href="/files/msgflux-1.0.3.tar.gz" data-yanked>yanked</a>
            <a href="/files/msgflux-1.0.4.tar.gz" data-requires-python="&gt;=99">future</a>
            <a href="/files/msgflux-1.1.0rc1.tar.gz">prerelease</a>
            <a href="/files/msgflux-1.0.%30.tar.gz?download=yes">encoded filename</a>
            <a>no file</a>
            </body></html>""",
        )

    original = httpx2.Client
    monkeypatch.setattr(
        updates.httpx2,
        "Client",
        lambda **kwargs: original(transport=httpx2.MockTransport(respond), **kwargs),
    )
    assert (
        updates.latest_version(
            prereleases=False, index_url="https://mirror.example/simple/"
        )
        == "1.0.1"
    )
    assert (
        updates.latest_version(
            prereleases=True, index_url="https://mirror.example/simple/"
        )
        == "1.1.0rc1"
    )
    assert len(requests) == 2
    assert "no-cache" in requests[0].headers["Cache-Control"]


@pytest.mark.parametrize(
    "url",
    [
        "",
        "/simple/",
        "file:///tmp/wheels",
        "ftp://host/simple",
        "https://host:bad/simple/",
        "https://host/simple?key=1",
        "https://host/simple#fragment",
    ],
)
def test_invalid_index_is_rejected_before_network(url, monkeypatch):
    client = Mock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(updates.httpx2, "Client", client)
    with pytest.raises(ValueError):
        updates.selected_index(url)
    client.assert_not_called()


def test_version_index_requires_check():
    with pytest.raises(SystemExit) as exited:
        cli.main(["version", "--index-url", "https://mirror.example/simple"])
    assert exited.value.code == 2


def test_version_check_uses_explicit_index(monkeypatch, capsys):
    remote = Mock(return_value="1.0.1")
    monkeypatch.setattr(updates, "latest_version", remote)
    cli.main(
        [
            "version",
            "--check",
            "--index-url",
            "https://mirror.example/private/simple",
            "--json",
        ]
    )
    assert (
        remote.call_args.kwargs["index_url"] == "https://mirror.example/private/simple/"
    )
    assert json.loads(capsys.readouterr().out)["latest"] == "1.0.1"


def test_receipt_default_index_used_and_explicit_index_overrides(installation):
    receipt = installation[1]
    receipt.write_text(
        receipt.read_text()
        + """
[tool.options]
index = [{url="https://mirror.example/simple", default=true, explicit=false, format="simple", authenticate="auto"}]
"""
    )
    assert updates.tool_installation().extras == ("coding", "service")
    assert updates.selected_index(None) == "https://mirror.example/simple/"
    assert (
        updates.selected_index("https://other.example/simple")
        == "https://other.example/simple/"
    )


def test_update_uses_same_index_for_query_and_installer(lifecycle, monkeypatch):
    events, options = lifecycle
    options.index_url = "https://mirror.example/team/simple"
    remote = Mock(return_value="1.0.1")
    monkeypatch.setattr(updates, "latest_version", remote)
    updates.update(options)
    assert remote.call_args.kwargs["index_url"] == "https://mirror.example/team/simple/"
    installer = events[2]
    assert installer[-2:] == ["--default-index", "https://mirror.example/team/simple/"]


@pytest.mark.parametrize(
    "index",
    [
        '[{url="https://one.example/", default=true}, {url="https://two.example/"}]',
        '[{url="https://one.example/", default=true, name="private"}]',
        '[{url="https://one.example/", default=false}]',
        '[{url="https://one.example/", default=true, authenticate="always"}]',
    ],
)
def test_unsupported_receipt_indexes_are_not_silently_dropped(installation, index):
    receipt = installation[1]
    receipt.write_text(receipt.read_text() + "\n[tool.options]\nindex = " + index)
    with pytest.raises(RuntimeError, match="manual"):
        updates.tool_installation()
