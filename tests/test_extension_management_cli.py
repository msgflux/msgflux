"""Manage installed extension selections without constructing or serving agents."""

import json
import tomllib

import pytest

from msgflux.cli import main
from msgflux.runtime.service.local.extensions_cli import _parse_options
from tests.test_extension_cli import installed_extension


def test_inspect_options_imports_factory_but_never_constructs(
    installed_extension, capsys
):
    config, package = installed_extension
    main(
        [
            "extensions",
            "inspect",
            "demo_test",
            "--options",
            "--config",
            str(config),
            "--json",
        ]
    )
    details = json.loads(capsys.readouterr().out)
    assert details["name"] == "demo_test"
    text = next(item for item in details["options"] if item["name"] == "text")
    assert text["has_default"] is True
    assert text["default"] == "hello"
    assert text["required"] is False
    assert (package / "imported").exists()
    assert not (package / "constructed").exists()
    main(["extensions", "inspect", "demo_test", "--options", "--config", str(config)])
    assert 'default="hello"' in capsys.readouterr().out


def test_add_remove_cli_preserves_comments_and_constructor_defaults(
    installed_extension, capsys
):
    config, package = installed_extension
    config.write_text(
        "# User configuration\n[defaults]\nagent='worker'\n"
        "[agents.worker] # retained\nfactory='unused_module:create'\n"
    )
    command = [
        "extensions",
        "add",
        "demo_test",
        "--agent",
        "worker",
        "--config",
        str(config),
        "--json",
    ]
    main(command)
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] and result["restart_required"]
    data = tomllib.loads(config.read_text())
    assert data["agents"]["worker"]["extensions"] == ["demo_test"]
    assert "extensions" not in data  # Do not copy defaults into global options.
    assert "# retained" in config.read_text()
    assert "# User configuration" in config.read_text()
    assert not (package / "constructed").exists()
    content = config.read_bytes()
    main(command)
    assert json.loads(capsys.readouterr().out)["changed"] is False
    assert config.read_bytes() == content
    main(
        [
            "extensions",
            "remove",
            "demo_test",
            "--agent",
            "worker",
            "--config",
            str(config),
            "--json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["changed"] is True
    assert tomllib.loads(config.read_text())["agents"]["worker"]["extensions"] == []
    assert not (package / "constructed").exists()


def test_options_alias_cli_keeps_other_agents_configuration(
    installed_extension, capsys
):
    config, _package = installed_extension
    config.write_text(
        config.read_text()
        + "\n[agents.other]\nfactory='unused_module:create'\nextensions=['demo_test']\n"
    )
    command = [
        "extensions",
        "add",
        "demo_test",
        "--agent",
        "worker",
        "--option",
        "text=Be brief.",
        "--config",
        str(config),
        "--json",
    ]
    original = config.read_bytes()
    with pytest.raises(SystemExit) as caught:
        main(command)
    assert caught.value.code == 1
    assert "--as" in json.loads(capsys.readouterr().out)["error"]
    assert config.read_bytes() == original
    main([*command, "--as", "worker_prompt"])
    result = json.loads(capsys.readouterr().out)
    assert result["extension"] == "worker_prompt"
    data = tomllib.loads(config.read_text())
    assert data["extensions"]["demo_test"]["options"]["text"] == "configured"
    assert data["extensions"]["worker_prompt"]["options"]["text"] == "Be brief."
    assert "worker_prompt" in data["agents"]["worker"]["extensions"]
    assert data["agents"]["other"]["extensions"] == ["demo_test"]


@pytest.mark.parametrize(
    "literal, expected",
    [
        ("true", True),
        ("15", 15),
        ("0.8", 0.8),
        ('"123"', "123"),
        ('["one", "two"]', ["one", "two"]),
        ("Be concise.", "Be concise."),
        ("", ""),
    ],
)
def test_option_literals_and_text(literal, expected):
    assert _parse_options([f"value={literal}"]) == {"value": expected}


@pytest.mark.parametrize(
    "values",
    [
        ["text"],
        ["nested.key=1"],
        ["text=one", "text=two"],
        ["text=[1,"],
        ['text="unterminated'],
        ["text=1\nother=2"],
    ],
)
def test_invalid_options_do_not_silently_change_meaning(values):
    with pytest.raises(ValueError):
        _parse_options(values)


@pytest.mark.parametrize(
    "arguments",
    [
        ["add", "demo_test"],
        ["remove", "demo_test"],
        ["ls", "--options"],
        ["inspect", "demo_test", "--option", "text=hi"],
        ["remove", "demo_test", "--agent", "worker", "--as", "other"],
    ],
)
def test_management_arguments_require_explicit_target(arguments):
    with pytest.raises(SystemExit) as caught:
        main(["extensions", *arguments])
    assert caught.value.code == 2
