"""Offline validation reuses manifest semantics without executing configured code."""

import json

import pytest

from msgflux import cli
from msgflux.runtime.service.local import config_cli

_VALID = """[defaults]
agent = "coding"
model = "unavailable-provider/model-id"
account = "not-logged-in"
accounts_dir = "missing-accounts"
[agents.coding]
preset = "coding"
"""


def test_validate_default_config_without_credentials_or_runtime(
    tmp_path, monkeypatch, capsys
):
    from msgflux.accounts.store import Accounts

    config = tmp_path / "config.toml"
    config.write_text(_VALID)
    monkeypatch.setattr(config_cli, "default_service_config_path", lambda: config)

    def unexpected_account(*args, **kwargs):
        raise AssertionError("validation must not open accounts")

    monkeypatch.setattr(Accounts, "__init__", unexpected_account)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    cli.main(["config", "validate", "--json"])
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result == {
        "valid": True,
        "config_path": str(config),
        "default_agent": "coding",
        "agents": ["coding"],
        "errors": [],
    }
    assert output.err == ""
    assert before == {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert config.read_text() == _VALID


def test_validate_custom_factory_does_not_import_user_code(
    tmp_path, monkeypatch, capsys
):
    marker = tmp_path / "imported"
    module = tmp_path / "unsafe_factory.py"
    module.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('imported')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='custom'\n[agents.custom]\nfactory='unsafe_factory:create'\n"
    )
    cli.main(["config", "validate", "--config", str(config)])
    assert "Configuration valid" in capsys.readouterr().out
    assert not marker.exists()


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[defaults", "Invalid service config"),
        (_VALID + "unexpected = true\n", "unknown field"),
        (_VALID + 'harness = "coding"\n', "unknown field"),
        (_VALID + 'factory = "unused:create"\n', "exactly one preset or factory"),
        (_VALID.replace('preset = "coding"', 'preset = ""'), "preset"),
        (_VALID.replace('preset = "coding"', 'preset = "unknown"'), "preset"),
        (_VALID.replace('agent = "coding"', 'agent = "missing"'), "declared agent"),
        (
            _VALID.replace(
                'model = "unavailable-provider/model-id"', 'model = "invalid"'
            ),
            "provider/model-id",
        ),
        (_VALID + 'permissions = ["invalid"]\n', "Invalid service config"),
        (_VALID + 'prompt_file = "missing.md"\n', "prompt_file"),
        (
            _VALID + '[agents.coding.tools]\nactive = ["missing"]\n',
            "Unknown active tool",
        ),
        (_VALID + 'subagents = ["unknown"]\n', "unknown subagent"),
        (
            _VALID
            + 'subagents = ["child"]\n[agents.child]\npreset = "coding"\nsubagents = ["coding"]\n',
            "cycle",
        ),
    ],
)
def test_invalid_config_json_is_structured_and_nonzero(
    tmp_path, capsys, body, expected
):
    config = tmp_path / "config.toml"
    config.write_text(body)
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", "--config", str(config), "--json"])
    assert exited.value.code == 1
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["valid"] is False and len(result["errors"]) == 1
    assert expected.lower() in result["errors"][0].lower()
    assert output.err == "" and "Traceback" not in output.out


def test_missing_config_does_not_initialize_directory(tmp_path, capsys):
    config = tmp_path / "missing" / "config.toml"
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", "--config", str(config)])
    assert exited.value.code == 1
    output = capsys.readouterr()
    assert not output.out and "No such file" in output.err
    assert not config.parent.exists()
