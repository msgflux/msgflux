"""Atomic extension edits preserve formatting and never construct user code."""

import os
import stat
import tomllib
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("tomlkit")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Config editing uses POSIX locks"
)

from msgflux.runtime.service.local.extension_config import mutate_extension_config


class Plugin:
    def __init__(self, label="default", window=8, enabled=True, settings=None):
        raise AssertionError("Configuration editing must never construct plugins")


class RequiredPlugin:
    def __init__(self, token):
        raise AssertionError("Configuration editing must never construct plugins")


def _config(tmp_path, *, selected="[]", other="[]", plugin="Plugin", options=""):
    path = tmp_path / "config.toml"
    path.write_text(
        "# My hand-edited configuration\n"
        "[defaults]\nagent = 'main'\n\n"
        "[agents.main]\nfactory = 'never_imported_agent:make' # keep factory comment\n"
        f"extensions = {selected} # keep selection comment\n\n"
        "[agents.other]\nfactory = 'never_imported_agent:make'\n"
        f"extensions = {other}\n\n"
        "[extensions.audit]\n"
        f"factory = 'tests.test_extension_config_editing:{plugin}' # keep plugin comment\n"
        f"{options}"
    )
    return path


def _mutate(path, **kwargs):
    return mutate_extension_config(
        path, name="audit", agent_id="main", operation="add", **kwargs
    )


def _read(path):
    return tomllib.loads(path.read_text())


def test_add_preserves_comments_inline_options_and_omits_constructor_defaults(tmp_path):
    path = _config(
        tmp_path, options="options = {label = 'original'} # inline options\n"
    )
    path.chmod(0o640)
    result = _mutate(path, options={"window": 4})
    assert result.changed and result.restart_required
    assert result.config_path == str(path.resolve())
    assert result.extension == "audit"
    config = _read(path)
    assert config["agents"]["main"]["extensions"] == ["audit"]
    assert config["extensions"]["audit"]["options"] == {
        "label": "original",
        "window": 4,
    }
    assert "enabled" not in config["extensions"]["audit"]["options"]
    text = path.read_text()
    for comment in (
        "# My hand-edited",
        "# keep factory",
        "# keep selection",
        "# keep plugin",
        "# inline options",
    ):
        assert comment in text
    assert "options = {" in text
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    before = path.read_bytes()
    inode = path.stat().st_ino
    again = _mutate(path, options={"window": 4})
    assert not again.changed and not again.restart_required
    assert path.read_bytes() == before
    assert path.stat().st_ino == inode


def test_add_without_options_never_writes_default_values(tmp_path):
    path = _config(tmp_path)
    _mutate(path)
    assert _read(path)["extensions"]["audit"] == {
        "factory": "tests.test_extension_config_editing:Plugin"
    }


@pytest.mark.parametrize("options", [None, {"unknown": 1}])
def test_invalid_or_missing_required_options_leave_original_unchanged(
    tmp_path, options
):
    path = _config(tmp_path, plugin="RequiredPlugin")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid Agent extension"):
        _mutate(path, options=options)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".config.toml.*.toml"))


def test_required_options_can_be_supplied_without_constructing_plugin(tmp_path):
    path = _config(tmp_path, plugin="RequiredPlugin")
    _mutate(path, options={"token": "configured"})
    assert _read(path)["extensions"]["audit"]["options"] == {"token": "configured"}


def test_shared_options_require_alias_and_alias_clones_without_modifying_source(
    tmp_path,
):
    path = _config(
        tmp_path,
        other="['audit']",
        options="options = {label = 'shared', window = 4}\n",
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--as"):
        _mutate(path, options={"window": 7})
    assert path.read_bytes() == before
    result = _mutate(path, alias="main-audit", options={"window": 7})
    assert result.extension == "main-audit"
    config = _read(path)
    assert config["agents"]["main"]["extensions"] == ["main-audit"]
    assert config["agents"]["other"]["extensions"] == ["audit"]
    assert config["extensions"]["audit"]["options"] == {"label": "shared", "window": 4}
    assert config["extensions"]["main-audit"] == {
        "factory": "tests.test_extension_config_editing:Plugin",
        "options": {"label": "shared", "window": 7},
    }
    assert not _mutate(path, alias="main-audit").changed


def test_alias_cannot_replace_an_existing_different_factory(tmp_path):
    path = _config(tmp_path)
    with path.open("a") as stream:
        stream.write("\n[extensions.other]\nfactory = 'different:Factory'\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="another factory"):
        _mutate(path, alias="other")
    assert path.read_bytes() == before


def test_remove_uninstalled_extension_retains_global_options_and_is_idempotent(
    tmp_path,
):
    path = _config(tmp_path, selected="['missing']")
    with path.open("a") as stream:
        stream.write("\n[extensions.missing.options]\nlabel = 'keep me'\n")
    result = mutate_extension_config(
        path, name="missing", agent_id="main", operation="remove"
    )
    assert result.changed
    config = _read(path)
    assert config["agents"]["main"]["extensions"] == []
    assert config["extensions"]["missing"]["options"] == {"label": "keep me"}
    assert not mutate_extension_config(
        path, name="missing", agent_id="main", operation="remove"
    ).changed


def test_remove_does_not_import_remaining_broken_factory(tmp_path):
    path = _config(tmp_path, selected="['audit']", plugin="UnavailablePlugin")
    result = mutate_extension_config(
        path, name="unused", agent_id="main", operation="remove"
    )
    assert not result.changed


def test_remove_repairs_shared_uninstalled_plugin_one_agent_at_a_time(tmp_path):
    path = _config(tmp_path, selected="['missing']", other="['missing']")
    with path.open("a") as stream:
        stream.write("\n[extensions.missing.options]\nlabel = 'preserved'\n")
    first = mutate_extension_config(
        path, name="missing", agent_id="main", operation="remove"
    )
    assert first.changed
    assert _read(path)["agents"]["main"]["extensions"] == []
    assert _read(path)["agents"]["other"]["extensions"] == ["missing"]
    second = mutate_extension_config(
        path, name="missing", agent_id="other", operation="remove"
    )
    assert second.changed
    assert _read(path)["agents"]["other"]["extensions"] == []
    assert _read(path)["extensions"]["missing"]["options"] == {"label": "preserved"}


def test_unknown_agent_and_missing_config_are_not_created(tmp_path):
    path = _config(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Unknown agent"):
        mutate_extension_config(path, name="audit", agent_id="missing", operation="add")
    assert path.read_bytes() == before
    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        _mutate(missing)
    assert not missing.exists()


def test_symlink_edits_canonical_target_and_preserves_symlink(tmp_path):
    path = _config(tmp_path)
    link = tmp_path / "linked.toml"
    link.symlink_to(path.name)
    result = _mutate(link)
    assert link.is_symlink()
    assert result.config_path == str(path.resolve())
    assert _read(path)["agents"]["main"]["extensions"] == ["audit"]
    assert (tmp_path / ".config.toml.lock").exists()
    assert not (tmp_path / ".linked.toml.lock").exists()


@pytest.mark.parametrize("replace_inode", [False, True])
def test_noncooperating_writer_is_detected_before_publish(
    tmp_path, monkeypatch, replace_inode
):
    from msgflux.runtime.service.local import extension_config

    path = _config(tmp_path)
    original = extension_config.load_service_manifest
    unrelated = path.read_bytes() + b"\n# external update\n"

    def validate_then_race(candidate):
        manifest = original(candidate)
        if replace_inode:
            replacement = tmp_path / "other.toml"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        else:
            path.write_bytes(unrelated)
        return manifest

    monkeypatch.setattr(extension_config, "load_service_manifest", validate_then_race)
    with pytest.raises(RuntimeError, match="changed while editing"):
        _mutate(path)
    if not replace_inode:
        assert path.read_bytes() == unrelated
    assert _read(path)["agents"]["main"]["extensions"] == []


def test_cooperating_concurrent_writers_do_not_lose_options(tmp_path):
    path = _config(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda options: _mutate(path, options=options),
                [{"label": "changed"}, {"window": 4}],
            )
        )
    assert all(result.changed for result in results)
    assert _read(path)["extensions"]["audit"]["options"] == {
        "label": "changed",
        "window": 4,
    }


@pytest.mark.parametrize("old,new", [(False, 0), (True, 1)])
def test_option_type_changes_are_not_lost_to_python_boolean_equality(
    tmp_path, old, new
):
    path = _config(tmp_path, options=f"options = {{enabled = {str(old).lower()}}}\n")
    result = _mutate(path, options={"enabled": new})
    assert result.changed
    actual = _read(path)["extensions"]["audit"]["options"]["enabled"]
    assert type(actual) is int
    assert actual == new


def test_nested_option_type_changes_are_preserved(tmp_path):
    path = _config(
        tmp_path, options="options = {settings = {flag = false, values = [true]}}\n"
    )
    assert _mutate(path, options={"settings": {"flag": 0, "values": [1]}}).changed
    actual = _read(path)["extensions"]["audit"]["options"]["settings"]
    assert type(actual["flag"]) is int
    assert type(actual["values"][0]) is int


def test_shared_option_guard_distinguishes_boolean_from_integer(tmp_path):
    path = _config(tmp_path, other="['audit']", options="options = {enabled = false}\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--as"):
        _mutate(path, options={"enabled": 0})
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "body",
    [
        "[defaults]\nagent='main'\n[agents]\nmain='bad'\n",
        "[defaults]\nagent='main'\n[agents.main]\nfactory='m:f'\nextensions='audit'\n",
        "[defaults]\nagent='main'\n[agents.main]\nfactory='m:f'\n[extensions.audit]\noptions='bad'\n",
        "extensions='bad'\n[defaults]\nagent='main'\n[agents.main]\nfactory='m:f'\n",
    ],
)
def test_malformed_config_shapes_fail_clearly_without_modification(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    with pytest.raises(ValueError, match="Invalid service config"):
        _mutate(path)
    assert path.read_text() == body


def test_missing_required_options_error_points_to_option_argument(tmp_path):
    path = _config(tmp_path, plugin="RequiredPlugin")
    with pytest.raises(ValueError, match=r"token.*--option KEY=VALUE"):
        _mutate(path)


def test_existing_alias_preserves_original_factory_quotes_on_noop(tmp_path):
    path = _config(tmp_path, selected="['same']")
    with path.open("a") as stream:
        stream.write(
            "\n[extensions.same]\nfactory = 'tests.test_extension_config_editing:Plugin'\n"
        )
    before = path.read_bytes()
    assert not _mutate(path, alias="same").changed
    assert path.read_bytes() == before


def test_concurrent_permission_change_is_not_overwritten(tmp_path, monkeypatch):
    from msgflux.runtime.service.local import extension_config

    path = _config(tmp_path)
    path.chmod(0o640)
    original = extension_config.load_service_manifest

    def validate_then_restrict(candidate):
        manifest = original(candidate)
        path.chmod(0o600)
        return manifest

    monkeypatch.setattr(
        extension_config, "load_service_manifest", validate_then_restrict
    )
    with pytest.raises(RuntimeError, match="changed while editing"):
        _mutate(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _read(path)["agents"]["main"]["extensions"] == []
