"""Offline CORS edits preserve the service configuration and require a restart."""

import json
import os
import stat
import tomllib

import pytest

from msgflux import cli
from msgflux.runtime.service.local import config_cli

pytest.importorskip("tomlkit")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Config editing uses POSIX locks"
)


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


def _invoke(path, capsys, action, *values):
    cli.main(["config", "cors", action, *values, "--config", str(path), "--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


@pytest.mark.parametrize(
    "origins", [None, ["https://example.com", "http://localhost:3000"]]
)
def test_list_reports_policy_without_modifying_configuration(tmp_path, capsys, origins):
    path = _config(tmp_path, origins)
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = _invoke(path, capsys, "ls")
    assert result == {
        "config_path": str(path.resolve()),
        "cors_allow_origins": origins or [],
    }
    assert path.read_bytes() == before and path.stat().st_ino == inode


def test_list_uses_default_service_configuration(tmp_path, monkeypatch, capsys):
    path = _config(tmp_path, ["*"])
    monkeypatch.setattr(config_cli, "default_service_config_path", lambda: path)
    cli.main(["config", "cors", "ls", "--json"])
    assert json.loads(capsys.readouterr().out) == {
        "config_path": str(path.resolve()),
        "cors_allow_origins": ["*"],
    }


def test_add_and_remove_preserve_comments_sections_and_file_mode(tmp_path, capsys):
    path = _config(tmp_path, ["http://localhost:3000"])
    path.chmod(0o640)
    result = _invoke(path, capsys, "add", "https://example.com")
    assert result == {
        "config_path": str(path.resolve()),
        "cors_allow_origins": ["http://localhost:3000", "https://example.com"],
        "changed": True,
        "restart_required": True,
    }
    added = tomllib.loads(path.read_text())
    assert added["http"]["cors_allow_origins"] == result["cors_allow_origins"]
    result = _invoke(path, capsys, "remove", "http://localhost:3000")
    assert result["cors_allow_origins"] == ["https://example.com"]
    assert result["changed"] and result["restart_required"]
    saved = path.read_text()
    for comment in (
        "# Hand-edited service",
        "# keep factory",
        "# browser policy",
        "# keep origins",
    ):
        assert comment in saved
    assert (
        tomllib.loads(saved)["agents"]["main"]["factory"] == "never_imported_agent:make"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.parametrize(
    "action, origins", [("add", ["https://example.com"]), ("remove", [])]
)
def test_idempotent_mutation_does_not_rewrite_file(tmp_path, capsys, action, origins):
    path = _config(tmp_path, origins)
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = _invoke(path, capsys, action, "https://example.com")
    assert result["cors_allow_origins"] == origins
    assert not result["changed"] and not result["restart_required"]
    assert path.read_bytes() == before and path.stat().st_ino == inode


@pytest.mark.parametrize(
    "origins, new_origin",
    [(["*"], "https://example.com"), (["https://example.com"], "*")],
)
def test_wildcard_cannot_be_silently_combined_or_replace_policy(
    tmp_path, capsys, origins, new_origin
):
    path = _config(tmp_path, origins)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exited:
        _invoke(path, capsys, "add", new_origin)
    assert exited.value.code == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "https://*.example.com",
        "https://example.com/path",
        "https://user:secret@example.com",
        "http://localhost:0",
    ],
)
def test_invalid_origin_never_changes_configuration(tmp_path, capsys, origin):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exited:
        _invoke(path, capsys, "add", origin)
    assert exited.value.code == 1
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".service.toml.*.toml"))


@pytest.mark.parametrize(
    "action, values",
    [
        ("ls", ()),
        ("add", ("https://example.com",)),
        ("remove", ("https://example.com",)),
    ],
)
def test_missing_configuration_is_not_created(tmp_path, capsys, action, values):
    path = tmp_path / "missing" / "service.toml"
    with pytest.raises(SystemExit) as exited:
        _invoke(path, capsys, action, *values)
    assert exited.value.code == 1
    assert not path.parent.exists()


def test_wildcard_add_remove_and_empty_policy_need_no_runtime(
    tmp_path, monkeypatch, capsys
):
    from msgflux.accounts.store import Accounts
    from msgflux.runtime.service.api import AgentService

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "CORS config commands must never start runtime or read credentials"
        )

    monkeypatch.setattr(Accounts, "__init__", forbidden)
    monkeypatch.setattr(AgentService, "__init__", forbidden)
    path = _config(tmp_path)
    added = _invoke(path, capsys, "add", "*")
    assert added["cors_allow_origins"] == ["*"] and added["changed"]
    removed = _invoke(path, capsys, "remove", "*")
    assert removed["cors_allow_origins"] == [] and removed["changed"]
    assert tomllib.loads(path.read_text())["http"]["cors_allow_origins"] == []


def test_cors_and_extension_edits_serialize_on_the_same_configuration(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from msgflux.runtime.service.local.cors_config import mutate_cors_config
    from msgflux.runtime.service.local.extension_config import mutate_extension_config

    path = _config(tmp_path, [])
    with path.open("a") as stream:
        stream.write(
            "\n[extensions.audit]\n"
            "factory = 'tests.test_extension_config_editing:Plugin'\n"
        )
    start = Barrier(2)

    def cors():
        start.wait(timeout=5)
        return mutate_cors_config(path, "https://example.com", operation="add")

    def extension():
        start.wait(timeout=5)
        return mutate_extension_config(
            path, name="audit", agent_id="main", operation="add"
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(cors), pool.submit(extension)]
        assert all(result.result(timeout=10).changed for result in pending)
    saved = tomllib.loads(path.read_text())
    assert saved["http"]["cors_allow_origins"] == ["https://example.com"]
    assert saved["agents"]["main"]["extensions"] == ["audit"]
    assert "# keep factory" in path.read_text()
    assert (tmp_path / ".service.toml.lock").exists()


def test_symlink_mutation_preserves_link_and_uses_canonical_lock(tmp_path, capsys):
    path = _config(tmp_path, [])
    link = tmp_path / "linked.toml"
    link.symlink_to(path.name)
    result = _invoke(link, capsys, "add", "https://example.com")
    assert link.is_symlink()
    assert result["config_path"] == str(path.resolve())
    assert tomllib.loads(path.read_text())["http"]["cors_allow_origins"] == [
        "https://example.com"
    ]
    assert (tmp_path / ".service.toml.lock").exists()
    assert not (tmp_path / ".linked.toml.lock").exists()


def test_human_output_distinguishes_restart_from_no_change(tmp_path, capsys):
    path = _config(tmp_path)
    args = ["config", "cors", "add", "https://example.com", "--config", str(path)]
    cli.main(args)
    first = capsys.readouterr()
    assert "https://example.com" in first.out
    assert "Restart the service" in first.out and first.err == ""
    cli.main(args)
    repeated = capsys.readouterr()
    assert "No change" in repeated.out
    assert "Restart" not in repeated.out and repeated.err == ""
    cli.main(["config", "cors", "remove", "https://example.com", "--config", str(path)])
    removed = capsys.readouterr()
    assert "CORS disabled" in removed.out and "Restart the service" in removed.out


@pytest.mark.parametrize(
    "arguments",
    [[], ["add"], ["remove"], ["ls", "https://example.com"], ["unknown"]],
)
def test_invalid_command_arguments_fail_before_editing(tmp_path, capsys, arguments):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "cors", *arguments, "--config", str(path)])
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "usage:" in captured.err
    assert path.read_bytes() == before


@pytest.mark.parametrize("command", [["validate"], ["cors", "ls"]])
def test_common_options_before_command_are_preserved(tmp_path, capsys, command):
    path = _config(tmp_path)
    cli.main(["config", "--config", str(path), "--json", *command])
    value = json.loads(capsys.readouterr().out)
    assert value["config_path"] == str(path.resolve())
    if command == ["validate"]:
        assert value["valid"]
    else:
        assert value["cors_allow_origins"] == []
