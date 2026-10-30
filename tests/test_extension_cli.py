"""Discovery reads installed distribution metadata without importing its package."""

import json
import sys
from pathlib import Path

import pytest

from msgflux.cli import main


@pytest.fixture
def installed_extension(tmp_path, monkeypatch):
    package = tmp_path / "demo_extension"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('imported').touch()\n"
        "from msgflux.nn import AgentExtension\n"
        "class Demo(AgentExtension):\n"
        "    def __init__(self, text='hello'):\n"
        "        super().__init__('original')\n"
        "        Path(__file__).with_name('constructed').touch()\n"
    )
    distribution = tmp_path / "demo_extension-0.1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo-extension\nVersion: 0.1.0\n"
    )
    (distribution / "entry_points.txt").write_text(
        "[msgflux.agent_extensions]\ndemo_test = demo_extension:Demo\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = tmp_path / "config.toml"
    config.write_text(
        "[defaults]\nagent='worker'\n"
        "[agents.worker]\nfactory='unused_module:create'\n"
        "extensions=['demo_test']\n"
        "[extensions.demo_test.options]\ntext='configured'\n"
    )
    yield config, package
    sys.modules.pop("demo_extension", None)


def test_extensions_list_and_inspect_real_distribution_metadata(
    installed_extension, capsys
):
    config, package = installed_extension
    main(["extensions", "ls", "--config", str(config), "--json"])
    items = json.loads(capsys.readouterr().out)
    info = next(item for item in items if item["name"] == "demo_test")
    assert info == {
        "name": "demo_test",
        "factory": "demo_extension:Demo",
        "distribution": "demo-extension",
        "version": "0.1.0",
    }
    main(["extensions", "inspect", "demo_test", "--config", str(config), "--json"])
    assert json.loads(capsys.readouterr().out) == info
    main(["extensions", "inspect", "demo_test", "--config", str(config)])
    assert "Package: demo-extension" in capsys.readouterr().out
    assert not (package / "imported").exists()
    assert not (package / "constructed").exists()


def test_config_validate_imports_selected_extension_without_constructing(
    installed_extension, capsys
):
    config, package = installed_extension
    main(["config", "validate", "--config", str(config), "--json"])
    assert json.loads(capsys.readouterr().out)["valid"] is True
    assert (package / "imported").exists()
    assert not (package / "constructed").exists()
    config.write_text(config.read_text().replace("text='configured'", "unknown=1"))
    with pytest.raises(SystemExit) as caught:
        main(["config", "validate", "--config", str(config), "--json"])
    assert caught.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False
    assert "demo_test" in result["errors"][0]
    assert not (package / "constructed").exists()


def test_extensions_inspect_unknown_reports_json_error(installed_extension, capsys):
    config, _package = installed_extension
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "extensions",
                "inspect",
                "not_installed",
                "--config",
                str(config),
                "--json",
            ]
        )
    assert caught.value.code == 1
    assert "not_installed" in json.loads(capsys.readouterr().out)["error"]


def test_extensions_list_without_config_uses_installed_metadata(
    installed_extension, tmp_path, monkeypatch, capsys
):
    from msgflux.runtime.service.local import extensions_cli

    monkeypatch.setattr(
        extensions_cli, "default_service_config_path", lambda: tmp_path / "absent.toml"
    )
    main(["extensions", "ls", "--json"])
    assert any(
        item["name"] == "demo_test" for item in json.loads(capsys.readouterr().out)
    )


@pytest.mark.parametrize("arguments", [["inspect"], ["ls", "unexpected"]])
def test_extensions_command_rejects_invalid_arguments(arguments):
    with pytest.raises(SystemExit) as caught:
        main(["extensions", *arguments])
    assert caught.value.code == 2


def test_discovery_works_before_model_or_agent_selection_is_valid(
    installed_extension, capsys
):
    config, package = installed_extension
    config.write_text(
        "[defaults]\nagent='coding'\n"
        "[agents.coding]\npreset='coding'\nextensions=['not_installed']\n"
    )
    main(["extensions", "ls", "--config", str(config), "--json"])
    assert any(
        item["name"] == "demo_test" for item in json.loads(capsys.readouterr().out)
    )
    assert not (package / "imported").exists()
