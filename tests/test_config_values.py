"""Regression tests for generic stored config-value reads and mutations.

These tests use the pinned backend API only and assert behaviour on a real
filesystem.  None of the factories named below exist; a correct backend must
validate the candidate manifest without importing or constructing them.
"""

import os
import stat
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("tomlkit")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Config editing uses POSIX locks"
)

from msgflux.runtime.service.local.config_values import (
    mutate_config_value,
    parse_config_path,
    parse_config_value,
    read_config_value,
)

BASE = """\
# hand-edited configuration
[defaults]
agent = "coding" # keep default agent comment
model = "unavailable-provider/model"

[agents.coding]
preset = "coding"

[agents.custom]
factory = "never_imported:create"

[extensions.tool]
factory = "never_imported_plugin:make"
"""


def _write(tmp_path, body=BASE):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path


def _read_doc(path):
    return tomllib.loads(path.read_text())


def _temp_files(tmp_path):
    return sorted(tmp_path.glob(".config.toml.*.toml"))


# --- parsing helpers -------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    [
        ("defaults.model", ("defaults", "model")),
        ("agents.coding.permissions", ("agents", "coding", "permissions")),
        ('agents."my.agent".factory', ("agents", "my.agent", "factory")),
        ('extensions.tool.options."a.b"', ("extensions", "tool", "options", "a.b")),
    ],
)
def test_parse_config_path(key, expected):
    assert parse_config_path(key) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", 42),
        ("1.5", 1.5),
        ("true", True),
        ("'hello'", "hello"),
        ('"hello"', "hello"),
        ("hello", "hello"),
        ("unavailable-provider/model", "unavailable-provider/model"),
        ("['read', 'execute']", ["read", "execute"]),
        ("{count = 1, label = 'x'}", {"count": 1, "label": "x"}),
    ],
)
def test_parse_config_value_typed_and_bare_strings(raw, expected):
    value = parse_config_value(raw)
    assert value == expected
    assert type(value) is type(expected)


@pytest.mark.parametrize(
    "raw",
    ["'unclosed", '"unclosed', "[1, 2", "{a = 1", "{a = }", "[1,, 2]"],
)
def test_parse_config_value_rejects_malformed_literals(raw):
    with pytest.raises(ValueError):
        parse_config_value(raw)


@pytest.mark.parametrize(
    "raw",
    ["1\ninjected = 2", "'x'\nfoo = 1"],
)
def test_parse_config_value_rejects_injected_assignments(raw):
    with pytest.raises(ValueError):
        parse_config_value(raw)


@pytest.mark.parametrize(
    "key",
    [
        "",
        ".",
        "defaults..model",
        ".defaults",
        "defaults.",
        "defaults model",
        "defaults[0].model",
        "defaults.model\ninjected = 1",
        "defaults.model=extra",
        "defaults.model # comment",
        "'unclosed",
        'defaults."unclosed',
    ],
)
def test_parse_config_path_rejects_malformed_keys(key):
    with pytest.raises(ValueError):
        parse_config_path(key)


# --- get: stored raw only --------------------------------------------------


def test_get_returns_stored_values_and_whole_document(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    result = read_config_value(path, "defaults.model")
    assert result.config_path == str(path.resolve())
    assert result.key == "defaults.model"
    assert result.value == "unavailable-provider/model"

    whole = read_config_value(path)
    assert whole.key is None
    assert whole.value["defaults"] == {
        "agent": "coding",
        "model": "unavailable-provider/model",
    }
    # Stored raw only: no effective defaults are materialised.
    assert "account" not in whole.value["defaults"]
    assert "reasoning_effort" not in whole.value["defaults"]
    assert path.read_bytes() == before


def test_get_reads_schema_invalid_but_syntactically_valid_config(tmp_path):
    path = _write(
        tmp_path, BASE.replace('model = "unavailable-provider/model"', "model = 123")
    )
    result = read_config_value(path, "defaults.model")
    assert result.value == 123
    assert type(result.value) is int


def test_get_missing_key_and_missing_file_do_not_create_files(tmp_path):
    path = _write(tmp_path)
    with pytest.raises(ValueError):
        read_config_value(path, "extensions.tool.options.absent")
    with pytest.raises(ValueError):
        read_config_value(path, "defaults.absent")

    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        read_config_value(missing, "defaults.model")
    assert not missing.exists()


# --- set: typed values, paths, parents -------------------------------------


def test_set_permissions_array_and_bare_model_path(tmp_path):
    path = _write(tmp_path)
    perms = mutate_config_value(
        path, "agents.coding.permissions", "['read', 'execute']", operation="set"
    )
    assert perms.changed and perms.restart_required
    assert perms.key == "agents.coding.permissions"
    assert _read_doc(path)["agents"]["coding"]["permissions"] == [
        "read",
        "execute",
    ]

    model = mutate_config_value(
        path, "defaults.model", "other-provider/model", operation="set"
    )
    assert model.changed
    assert read_config_value(path, "defaults.model").value == ("other-provider/model")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("42", 42),
        ("1.5", 1.5),
        ("false", False),
        ("'quoted'", "quoted"),
        ("bare", "bare"),
        ("['a', 'b']", ["a", "b"]),
    ],
)
def test_set_typed_values_in_extension_options(tmp_path, raw, expected):
    path = _write(tmp_path)
    result = mutate_config_value(
        path, "extensions.tool.options.sample", raw, operation="set"
    )
    assert result.changed and result.restart_required
    value = read_config_value(path, "extensions.tool.options.sample").value
    assert value == expected
    assert type(value) is type(expected)


def test_set_quoted_dotted_agent_id_and_extension_option(tmp_path):
    path = _write(tmp_path)
    agent = mutate_config_value(
        path, 'agents."my.agent".factory', "'never_imported:create'", operation="set"
    )
    assert agent.changed
    assert _read_doc(path)["agents"]["my.agent"]["factory"] == ("never_imported:create")

    option = mutate_config_value(
        path, 'extensions.tool.options."a.b"', "'dotted'", operation="set"
    )
    assert option.changed
    assert read_config_value(path, 'extensions.tool.options."a.b"').value == ("dotted")


def test_set_inline_table_and_creates_missing_table_parents(tmp_path):
    path = _write(tmp_path)
    whole = mutate_config_value(
        path,
        "extensions.tool.options",
        "{count = 2, nested = {flag = true}}",
        operation="set",
    )
    assert whole.changed
    assert read_config_value(path, "extensions.tool.options").value == {
        "count": 2,
        "nested": {"flag": True},
    }

    created = mutate_config_value(
        path,
        "extensions.tool.options.created",
        "['https://example.invalid']",
        operation="set",
    )
    assert created.changed
    assert read_config_value(path, "extensions.tool.options.created").value == [
        "https://example.invalid"
    ]


def test_set_creates_missing_options_table_and_http_parent(tmp_path):
    path = _write(tmp_path)
    assert "options" not in _read_doc(path)["extensions"]["tool"]
    mutate_config_value(path, "extensions.tool.options.count", "3", operation="set")
    assert _read_doc(path)["extensions"]["tool"]["options"] == {"count": 3}

    mutate_config_value(
        path, "http.cors_allow_origins", "['http://localhost:3000']", operation="set"
    )
    assert _read_doc(path)["http"]["cors_allow_origins"] == ["http://localhost:3000"]


# --- rejection / unchanged file --------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["", "defaults..model", "defaults.model\nx=1", "defaults.model # c"],
)
def test_mutate_rejects_malformed_paths_unchanged(tmp_path, key):
    path = _write(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, key, "'x'", operation="set")
    assert path.read_bytes() == before
    assert _temp_files(tmp_path) == []


@pytest.mark.parametrize("raw", ["'unclosed", "[1,", "{a=}", "1\ninjected=2"])
def test_mutate_rejects_malformed_values_unchanged(tmp_path, raw):
    path = _write(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, "extensions.tool.options.bad", raw, operation="set")
    assert path.read_bytes() == before
    assert _temp_files(tmp_path) == []


def test_mutate_rejects_unknown_field_and_bad_permissions(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, "agents.coding.unknown_field", "'x'", operation="set")
    with pytest.raises(ValueError):
        mutate_config_value(
            path, "agents.coding.permissions", "['read', 'bogus']", operation="set"
        )
    assert path.read_bytes() == before
    assert _temp_files(tmp_path) == []


def test_mutate_rejects_scalar_traversal(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, "defaults.agent.child", "'x'", operation="set")
    with pytest.raises(ValueError):
        mutate_config_value(path, "defaults.agent.child", operation="unset")
    assert path.read_bytes() == before


def test_invalid_original_toml_cannot_be_mutated(tmp_path):
    path = _write(tmp_path, "[defaults]\nagent = \n")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, "defaults.model", "'other/model'", operation="set")
    assert path.read_bytes() == before


def test_mutate_missing_file_is_not_created(tmp_path):
    missing = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        mutate_config_value(missing, "defaults.model", "'other/model'", operation="set")
    assert not missing.exists()


# --- idempotence, preservation, type sensitivity ---------------------------


def test_idempotent_set_preserves_inode_comments_and_mode(tmp_path):
    path = _write(tmp_path)
    path.chmod(0o640)
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = mutate_config_value(
        path, "defaults.model", "'unavailable-provider/model'", operation="set"
    )
    assert not result.changed
    assert not result.restart_required
    assert path.read_bytes() == before
    assert path.stat().st_ino == inode
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert b"# hand-edited" in path.read_bytes()
    assert b"# keep default agent comment" in path.read_bytes()


def test_bool_vs_int_distinguishes_nested_values(tmp_path):
    path = _write(tmp_path)
    mutate_config_value(
        path,
        "extensions.tool.options.settings",
        "{flag = false, values = [true]}",
        operation="set",
    )
    changed = mutate_config_value(
        path,
        "extensions.tool.options.settings",
        "{flag = 0, values = [1]}",
        operation="set",
    )
    assert changed.changed
    settings = read_config_value(path, "extensions.tool.options.settings").value
    assert type(settings["flag"]) is int
    assert type(settings["values"][0]) is int

    again = mutate_config_value(
        path,
        "extensions.tool.options.settings",
        "{flag = 0, values = [1]}",
        operation="set",
    )
    assert not again.changed


# --- unset ------------------------------------------------------------------


def test_unset_explicit_option_restores_inheritance(tmp_path):
    path = _write(tmp_path)
    mutate_config_value(
        path, "agents.coding.model", "other-provider/model", operation="set"
    )
    assert read_config_value(path, "agents.coding.model").value == (
        "other-provider/model"
    )
    result = mutate_config_value(path, "agents.coding.model", operation="unset")
    assert result.changed and result.restart_required
    doc = _read_doc(path)
    assert "model" not in doc["agents"]["coding"]
    assert doc["defaults"]["model"] == "unavailable-provider/model"
    # No defaults were materialised into the agent table.
    assert doc["agents"]["coding"] == {"preset": "coding"}


def test_unset_missing_is_idempotent_and_preserves_file(tmp_path):
    path = _write(tmp_path)
    path.chmod(0o640)
    before = path.read_bytes()
    inode = path.stat().st_ino
    result = mutate_config_value(
        path, "extensions.tool.options.absent", operation="unset"
    )
    assert not result.changed
    assert not result.restart_required
    assert path.read_bytes() == before
    assert path.stat().st_ino == inode
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_unset_missing_tables_is_noop_without_creating_them(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    for key in (
        "http.cors_allow_origins",
        "extensions.tool.options.absent",
    ):
        result = mutate_config_value(path, key, operation="unset")
        assert not result.changed
    assert path.read_bytes() == before
    doc = _read_doc(path)
    assert "http" not in doc
    assert "options" not in doc["extensions"]["tool"]


def test_unset_required_values_is_rejected_unchanged(tmp_path):
    path = _write(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        mutate_config_value(path, "defaults.agent", operation="unset")
    with pytest.raises(ValueError):
        mutate_config_value(path, "defaults.model", operation="unset")
    assert path.read_bytes() == before
    assert _temp_files(tmp_path) == []


# --- repair, concurrency, offline ------------------------------------------


def test_repair_wrong_schema_type(tmp_path):
    body = BASE.replace('model = "unavailable-provider/model"', "model = 123")
    path = _write(tmp_path, body)
    result = mutate_config_value(
        path, "defaults.model", "unavailable-provider/model", operation="set"
    )
    assert result.changed
    assert read_config_value(path, "defaults.model").value == (
        "unavailable-provider/model"
    )


def test_repair_obsolete_unknown_field_by_unsetting_it(tmp_path):
    body = BASE.replace(
        'preset = "coding"',
        'preset = "coding"\nobsolete_field = "x"',
    )
    path = _write(tmp_path, body)
    result = mutate_config_value(
        path, "agents.coding.obsolete_field", operation="unset"
    )
    assert result.changed
    assert _read_doc(path)["agents"]["coding"] == {"preset": "coding"}


def test_repair_wrong_permissions_type_by_setting_it(tmp_path):
    body = BASE.replace('preset = "coding"', 'preset = "coding"\npermissions = "read"')
    path = _write(tmp_path, body)
    result = mutate_config_value(
        path, "agents.coding.permissions", "['read']", operation="set"
    )
    assert result.changed
    assert _read_doc(path)["agents"]["coding"]["permissions"] == ["read"]


def test_concurrent_independent_edits_preserve_both_values(tmp_path):
    path = _write(tmp_path)

    def edit(item):
        key, raw = item
        return mutate_config_value(path, key, raw, operation="set")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                edit,
                [
                    ("extensions.tool.options.first", "'one'"),
                    ("extensions.tool.options.second", "2"),
                ],
            )
        )
    assert all(result.changed for result in results)
    options = read_config_value(path, "extensions.tool.options").value
    assert options == {"first": "one", "second": 2}


def test_mutation_never_imports_factories(tmp_path):
    assert "never_imported" not in sys.modules
    assert "never_imported_plugin" not in sys.modules
    path = _write(tmp_path)
    mutate_config_value(path, "extensions.tool.options.count", "1", operation="set")
    mutate_config_value(
        path, "agents.custom.factory", "'never_imported:create'", operation="set"
    )
    assert "never_imported" not in sys.modules
    assert "never_imported_plugin" not in sys.modules


@pytest.mark.parametrize("raw,expected", [("true # comment", True), ("1 # comment", 1)])
def test_toml_value_comments_match_extension_option_semantics(raw, expected):
    assert parse_config_value(raw) == expected


@pytest.mark.parametrize(
    "key", ["defaults.model = 0 #", "defaults.model = 0 # discarded"]
)
def test_config_key_rejects_assignment_smuggling(key):
    with pytest.raises(ValueError):
        parse_config_path(key)
