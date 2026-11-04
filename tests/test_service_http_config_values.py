"""HTTP config values stay offline, validated, and restart-gated.

The specialised CORS commands are gone: browser origins are now ordinary
stored values edited through the generic ``config get/set/unset`` API and
``config_values`` backend.  These tests focus on HTTP semantics and shared
editor compatibility; generic value/CLI behaviour lives in the
``test_config_values*`` suites.
"""

import json
import os
import stat
import tomllib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from msgflux import cli
from msgflux.runtime.service.http import ServiceHTTPConfig
from msgflux.runtime.service.local import config_cli
from msgflux.runtime.service.local.config_values import (
    mutate_config_value,
    read_config_value,
)
from msgflux.runtime.service.local.manifest import load_service_manifest

pytest.importorskip("tomlkit")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Config editing uses POSIX locks"
)

KEY = "http.cors_allow_origins"
ORIGIN = "https://example.com"
LOCAL = "http://localhost:3000"


def _config(tmp_path, origins=None):
    path = tmp_path / "service.toml"
    text = (
        "# Hand-edited service\n"
        "[defaults]\nagent = 'main'\n\n"
        "[agents.main]\n"
        "factory = 'never_imported_agent:make' # keep factory\n"
    )
    if origins is not None:
        text += (
            "\n[http] # browser policy\n"
            f"cors_allow_origins = {json.dumps(origins)} # keep origins\n"
        )
    path.write_text(text)
    return path


def _invoke(path, capsys, *command):
    cli.main(["config", *command, "--config", str(path), "--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _get(path, capsys):
    return _invoke(path, capsys, "get", KEY)


def _set(path, capsys, origins):
    return _invoke(path, capsys, "set", KEY, json.dumps(origins))


def _unset(path, capsys):
    return _invoke(path, capsys, "unset", KEY)


def _set_error(path, capsys, origins):
    with pytest.raises(SystemExit) as exited:
        _invoke(path, capsys, "set", KEY, json.dumps(origins))
    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["error"]


# --- get: stored raw only ---------------------------------------------------


def test_get_returns_stored_origins_without_modifying_configuration(tmp_path, capsys):
    path = _config(tmp_path, [ORIGIN, LOCAL])
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = _get(path, capsys)
    assert result == {
        "config_path": str(path.resolve()),
        "key": KEY,
        "value": [ORIGIN, LOCAL],
    }
    assert path.read_bytes() == before and path.stat().st_ino == inode


def test_get_absent_field_raises_and_whole_config_shows_absence(tmp_path, capsys):
    path = _config(tmp_path)
    before = path.read_bytes()
    # Stored raw only: the service default (empty tuple) is not materialised.
    with pytest.raises(ValueError):
        read_config_value(path, KEY)
    with pytest.raises(SystemExit) as exited:
        _get(path, capsys)
    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().out)["error"]
    whole = _invoke(path, capsys, "get")
    assert "http" not in whole["value"]
    assert path.read_bytes() == before


def test_get_uses_default_service_configuration(tmp_path, monkeypatch, capsys):
    path = _config(tmp_path, ["*"])
    monkeypatch.setattr(config_cli, "default_service_config_path", lambda: path)
    cli.main(["config", "get", KEY, "--json"])
    assert json.loads(capsys.readouterr().out) == {
        "config_path": str(path.resolve()),
        "key": KEY,
        "value": ["*"],
    }


# --- set/unset: exact values, metadata, preservation ------------------------


def test_set_replaces_policy_preserves_comments_and_file_mode(tmp_path, capsys):
    path = _config(tmp_path, [LOCAL])
    path.chmod(0o640)
    result = _set(path, capsys, [LOCAL, ORIGIN])
    # Mutations return metadata only; they never echo the assigned value.
    assert result == {
        "config_path": str(path.resolve()),
        "key": KEY,
        "changed": True,
        "restart_required": True,
    }
    saved = tomllib.loads(path.read_text())
    assert saved["http"]["cors_allow_origins"] == [LOCAL, ORIGIN]
    assert saved["agents"]["main"]["factory"] == "never_imported_agent:make"
    for comment in (
        "# Hand-edited service",
        "# keep factory",
        "# browser policy",
        "# keep origins",
    ):
        assert comment in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_set_same_value_and_unset_absent_field_are_idempotent(tmp_path, capsys):
    path = _config(tmp_path, [ORIGIN])
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = _set(path, capsys, [ORIGIN])
    assert not result["changed"] and not result["restart_required"]
    assert path.read_bytes() == before and path.stat().st_ino == inode

    absent = tmp_path / "absent.toml"
    absent.write_text(
        "# Hand-edited service\n[defaults]\nagent = 'main'\n\n"
        "[agents.main]\nfactory = 'never_imported_agent:make'\n"
    )
    before_absent = absent.read_bytes()
    result = _unset(absent, capsys)
    assert not result["changed"] and not result["restart_required"]
    assert absent.read_bytes() == before_absent
    assert "http" not in tomllib.loads(absent.read_text())


def test_unset_removes_field_and_returns_to_service_default(tmp_path, capsys):
    path = _config(tmp_path, ["*"])
    result = _unset(path, capsys)
    assert result == {
        "config_path": str(path.resolve()),
        "key": KEY,
        "changed": True,
        "restart_required": True,
    }
    assert "cors_allow_origins" not in tomllib.loads(path.read_text())["http"]
    assert load_service_manifest(path).http == ServiceHTTPConfig()
    with pytest.raises(ValueError):
        read_config_value(path, KEY)


def test_set_replaces_wildcard_with_explicit_list(tmp_path, capsys):
    path = _config(tmp_path, ["*"])
    result = _set(path, capsys, [ORIGIN])
    assert result["changed"] and result["restart_required"]
    assert _get(path, capsys)["value"] == [ORIGIN]
    assert load_service_manifest(path).http.cors_allow_origins == (ORIGIN,)


# --- validation before publication ------------------------------------------


@pytest.mark.parametrize(
    "origins",
    [
        ["*", ORIGIN],
        [ORIGIN, "*"],
        ["*", "*"],
        ["null"],
        ["https://*.example.com"],
        ["https://example.com/path"],
        ["https://user:secret@example.com"],
        ["http://localhost:0"],
        ["ftp://example.com"],
        [""],
    ],
)
def test_invalid_origins_never_change_configuration(tmp_path, capsys, origins):
    path = _config(tmp_path, [LOCAL])
    before = path.read_bytes()
    _set_error(path, capsys, origins)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".service.toml.*.toml"))


@pytest.mark.parametrize(
    "command",
    [
        ("get", KEY),
        ("set", KEY, json.dumps([ORIGIN])),
        ("unset", KEY),
    ],
)
def test_missing_configuration_is_not_created(tmp_path, capsys, command):
    path = tmp_path / "missing" / "service.toml"
    with pytest.raises(SystemExit) as exited:
        _invoke(path, capsys, *command)
    assert exited.value.code == 1
    assert not path.parent.exists()


def test_generic_edits_need_no_runtime(tmp_path, monkeypatch, capsys):
    from msgflux.accounts.store import Accounts
    from msgflux.runtime.service.api import AgentService

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Config commands must never start runtime or read credentials"
        )

    monkeypatch.setattr(Accounts, "__init__", forbidden)
    monkeypatch.setattr(AgentService, "__init__", forbidden)
    path = _config(tmp_path)
    assert _set(path, capsys, ["*"])["changed"]
    assert _unset(path, capsys)["changed"]
    assert _set(path, capsys, [ORIGIN])["changed"]
    assert _get(path, capsys)["value"] == [ORIGIN]


# --- shared editor compatibility --------------------------------------------


def test_http_and_extension_edits_serialize_on_the_same_configuration(tmp_path):
    from msgflux.runtime.service.local.extension_config import (
        mutate_extension_config,
    )

    path = _config(tmp_path, [])
    with path.open("a") as stream:
        stream.write(
            "\n[extensions.audit]\n"
            "factory = 'tests.test_extension_config_editing:Plugin'\n"
        )
    start = Barrier(2)

    def http_edit():
        start.wait(timeout=5)
        return mutate_config_value(path, KEY, json.dumps([ORIGIN]), operation="set")

    def extension_edit():
        start.wait(timeout=5)
        return mutate_extension_config(
            path, name="audit", agent_id="main", operation="add"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(http_edit), pool.submit(extension_edit)]
        assert all(result.result(timeout=10).changed for result in pending)
    saved = tomllib.loads(path.read_text())
    assert saved["http"]["cors_allow_origins"] == [ORIGIN]
    assert saved["agents"]["main"]["extensions"] == ["audit"]
    assert "# keep factory" in path.read_text()
    assert (tmp_path / ".service.toml.lock").exists()


def test_symlink_mutation_preserves_link_and_uses_canonical_lock(tmp_path, capsys):
    path = _config(tmp_path, [])
    link = tmp_path / "linked.toml"
    link.symlink_to(path.name)
    result = _set(link, capsys, [ORIGIN])
    assert link.is_symlink()
    assert result["config_path"] == str(path.resolve())
    assert tomllib.loads(path.read_text())["http"]["cors_allow_origins"] == [ORIGIN]
    assert (tmp_path / ".service.toml.lock").exists()
    assert not (tmp_path / ".linked.toml.lock").exists()


def test_human_output_distinguishes_restart_from_no_change(tmp_path, capsys):
    path = _config(tmp_path)
    set_args = ["config", "set", KEY, json.dumps([ORIGIN]), "--config", str(path)]
    cli.main(set_args)
    first = capsys.readouterr()
    assert KEY in first.out
    assert "Restart the service" in first.out and first.err == ""
    cli.main(set_args)
    repeated = capsys.readouterr()
    assert "No change" in repeated.out
    assert "Restart" not in repeated.out and repeated.err == ""
    cli.main(["config", "unset", KEY, "--config", str(path)])
    removed = capsys.readouterr()
    assert "Removed" in removed.out and "Restart the service" in removed.out
    cli.main(["config", "unset", KEY, "--config", str(path)])
    assert "No change" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["set"],
        ["set", KEY],
        ["unset", KEY, "extra"],
        ["get", KEY, "extra"],
        ["unknown"],
    ],
)
def test_invalid_command_arguments_fail_before_editing(tmp_path, capsys, arguments):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", *arguments, "--config", str(path)])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage:" in captured.err
    assert path.read_bytes() == before


@pytest.mark.parametrize("command", [["get", KEY], ["set", KEY, json.dumps(["*"])]])
def test_common_options_before_command_are_preserved(tmp_path, capsys, command):
    path = _config(tmp_path, [])
    cli.main(["config", "--config", str(path), "--json", *command])
    value = json.loads(capsys.readouterr().out)
    assert value["config_path"] == str(path.resolve())
    assert value["key"] == KEY
