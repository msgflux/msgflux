from pathlib import Path

import pytest

from msgflux.vulcano import KeyBindings, VulcanoSettings


def test_settings_merge_global_and_project_toml(tmp_path):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    home.mkdir()
    (cwd / ".vulcano").mkdir(parents=True)
    (home / "config.toml").write_text(
        """
[ui.editor]
min_height = 4
max_height = 12

[ui.transcript]
mode = "compact"

[ui.keybindings]
command_palette = "ctrl+k"
newline = ["shift+enter"]

[sessions]
max_tabs = 7
""".strip(),
        encoding="utf-8",
    )
    project_config = cwd / ".vulcano" / "config.toml"
    project_config.write_text(
        """
[ui.editor]
max_height = 18

[ui.keybindings]
follow_up = ["ctrl+enter"]

[sessions]
max_tabs = 5
""".strip(),
        encoding="utf-8",
    )

    settings = VulcanoSettings.load(cwd=cwd, home=home)

    assert settings.home == home.resolve()
    assert settings.cwd == cwd.resolve()
    assert settings.editor.min_height == 4
    assert settings.editor.max_height == 18
    assert settings.transcript.mode == "compact"
    assert settings.keybindings.keys("command_palette") == ("ctrl+k",)
    assert settings.keybindings.keys("follow_up") == ("ctrl+enter",)
    assert settings.keybindings.keys("newline") == ("shift+enter",)
    assert settings.keybindings.keys("cancel") == ("escape",)
    assert settings.keybindings.keys("toggle_sidebar") == ("alt+s",)
    assert settings.keybindings.keys("session_prefix") == ("ctrl+g",)
    assert settings.sessions.max_tabs == 5
    assert settings.sources == (
        home / "config.toml",
        project_config,
    )


def test_settings_honor_vulcano_home_environment(tmp_path, monkeypatch):
    home = tmp_path / "custom-vulcano"
    monkeypatch.setenv("VULCANO_HOME", str(home))

    settings = VulcanoSettings.defaults(cwd=tmp_path)

    assert settings.home == home.resolve()


def test_keybindings_reject_collisions():
    with pytest.raises(ValueError, match="assigned to both"):
        KeyBindings.from_mapping(
            {
                "cancel": "ctrl+x",
                "quit": "ctrl+x",
            }
        )


def test_session_prefix_can_be_reconfigured():
    keybindings = KeyBindings.from_mapping({"session_prefix": "ctrl+a"})

    assert keybindings.keys("session_prefix") == ("ctrl+a",)


def test_invalid_editor_height_is_rejected(tmp_path):
    config = tmp_path / ".vulcano" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "[ui.editor]\nmin_height = 8\nmax_height = 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_height"):
        VulcanoSettings.load(cwd=tmp_path, home=tmp_path / "home")


def test_invalid_transcript_mode_is_rejected(tmp_path):
    config = tmp_path / ".vulcano" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        '[ui.transcript]\nmode = "dense"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"transcript\.mode"):
        VulcanoSettings.load(cwd=tmp_path, home=tmp_path / "home")


@pytest.mark.parametrize("value", [0, -1, 11, True])
def test_invalid_session_tab_limit_is_rejected(tmp_path, value):
    config = tmp_path / ".vulcano" / "config.toml"
    config.parent.mkdir()
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    config.write_text(
        f"[sessions]\nmax_tabs = {rendered}\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=r"sessions\.max_tabs"):
        VulcanoSettings.load(cwd=tmp_path, home=tmp_path / "home")


def test_settings_paths_are_path_objects(tmp_path):
    settings = VulcanoSettings.defaults(cwd=tmp_path, home=tmp_path / "home")

    assert isinstance(settings.cwd, Path)
    assert isinstance(settings.home, Path)
    assert settings.transcript.mode == "full"
    assert settings.sessions.max_tabs == 5


def test_session_tab_limit_accepts_ten(tmp_path):
    config = tmp_path / ".vulcano" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "[sessions]\nmax_tabs = 10\n",
        encoding="utf-8",
    )

    settings = VulcanoSettings.load(cwd=tmp_path, home=tmp_path / "home")

    assert settings.sessions.max_tabs == 10
