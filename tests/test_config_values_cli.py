"""Generic configuration commands stay offline and edit only validated values."""

import json
import tomllib

import pytest

from msgflux import cli
from msgflux.runtime.service.local import config_cli

pytest.importorskip("tomlkit")


def _config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "# Personal configuration\n"
        "[defaults]\nagent = 'coding'\nmodel = 'provider/model'\n"
        "[agents.coding]\npreset = 'coding' # preserve comment\n"
        "[extensions.demo]\nfactory = 'never_imported:make'\n"
    )
    return path


def _invoke(path, capsys, *command):
    cli.main(["config", *command, "--config", str(path), "--json"])
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_get_default_document_and_bare_string_output(tmp_path, monkeypatch, capsys):
    path = _config(tmp_path)
    monkeypatch.setattr(config_cli, "default_service_config_path", lambda: path)
    before = path.read_bytes()
    cli.main(["config", "get", "defaults.model"])
    assert capsys.readouterr().out == "provider/model\n"
    cli.main(["config", "get", "--json"])
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "config_path": str(path.resolve()),
        "key": None,
        "value": tomllib.loads(before.decode()),
    }
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".*.lock"))


@pytest.mark.parametrize("prefix", [True, False])
def test_common_flags_work_before_or_after_action(tmp_path, capsys, prefix):
    path = _config(tmp_path)
    common = ["--config", str(path), "--json"]
    args = (
        [*common, "get", "defaults.agent"]
        if prefix
        else ["get", "defaults.agent", *common]
    )
    cli.main(["config", *args])
    assert json.loads(capsys.readouterr().out)["value"] == "coding"


def test_set_permissions_and_unset_restore_config_inheritance(tmp_path, capsys):
    path = _config(tmp_path)
    mutation = _invoke(
        path, capsys, "set", "agents.coding.permissions", '["read", "execute"]'
    )
    assert mutation == {
        "config_path": str(path.resolve()),
        "key": "agents.coding.permissions",
        "changed": True,
        "restart_required": True,
    }
    assert _invoke(path, capsys, "get", "agents.coding.permissions")["value"] == [
        "read",
        "execute",
    ]
    assert not _invoke(
        path, capsys, "set", "agents.coding.permissions", '["read", "execute"]'
    )["changed"]
    assert _invoke(path, capsys, "unset", "agents.coding.permissions")["changed"]
    assert "permissions" not in tomllib.loads(path.read_text())["agents"]["coding"]
    assert "# preserve comment" in path.read_text()


@pytest.mark.parametrize(
    "raw,expected",
    [("false", False), ("42", 42), ('"42"', "42"), ("plain text", "plain text")],
)
def test_typed_value_output_and_json(tmp_path, capsys, raw, expected):
    path = _config(tmp_path)
    _invoke(path, capsys, "set", "extensions.demo.options.value", raw)
    value = _invoke(path, capsys, "get", "extensions.demo.options.value")["value"]
    assert value == expected and type(value) is type(expected)
    cli.main(["config", "get", "extensions.demo.options.value", "--config", str(path)])
    assert capsys.readouterr().out.strip() == (
        expected if isinstance(expected, str) else json.dumps(expected)
    )


def test_mutation_output_does_not_echo_values_or_restart_service(
    tmp_path, monkeypatch, capsys
):
    from msgflux.accounts import Accounts

    path = _config(tmp_path)
    monkeypatch.setattr(
        Accounts, "__init__", lambda *_args, **_kwargs: pytest.fail("account read")
    )
    cli.main(
        [
            "config",
            "set",
            "extensions.demo.options.token",
            "private-placeholder",
            "--config",
            str(path),
        ]
    )
    output = capsys.readouterr().out
    assert "Updated" in output and "Restart the service" in output
    assert "private-placeholder" not in output
    assert not (tmp_path / ".msgflux").exists()
    assert "No change" not in output
    assert not _invoke(path, capsys, "unset", "extensions.demo.options.absent")[
        "restart_required"
    ]


@pytest.mark.parametrize(
    "command",
    [
        ("get", "missing.key"),
        ("set", "agents.coding.permissions", '["invalid"]'),
        ("set", "defaults.model", "[oops"),
        ("unset", "defaults.agent"),
    ],
)
def test_value_errors_are_json_and_do_not_publish(tmp_path, capsys, command):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.main(["config", *command, "--config", str(path), "--json"])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["error"]
    assert path.read_bytes() == before


def test_missing_config_text_error_and_usage_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["config", "get", "--config", str(tmp_path / "absent.toml")])
    assert error.value.code == 1
    assert "Error:" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:
        cli.main(["config", "set", "defaults.model"])
    assert usage.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_whole_document_text_is_readable_and_specialized_cors_command_is_removed(
    tmp_path, capsys
):
    path = _config(tmp_path)
    cli.main(["config", "get", "--config", str(path)])
    output = capsys.readouterr().out
    assert json.loads(output) == tomllib.loads(path.read_text())
    assert '\n  "defaults"' in output
    before = path.read_bytes()
    with pytest.raises(SystemExit) as removed:
        cli.main(["config", "cors", "ls", "--config", str(path)])
    assert removed.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert path.read_bytes() == before
