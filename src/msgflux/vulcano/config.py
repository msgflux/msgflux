from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import tomllib

__all__ = [
    "DEFAULT_KEY_BINDINGS",
    "EditorSettings",
    "KeyBindings",
    "VulcanoSettings",
]


DEFAULT_KEY_BINDINGS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "submit": ("enter",),
        "newline": ("shift+enter", "ctrl+j"),
        "follow_up": ("alt+enter",),
        "cancel": ("escape",),
        "clear": ("ctrl+l",),
        "quit": ("ctrl+c",),
        "command_palette": ("ctrl+p",),
    }
)


@dataclass(frozen=True)
class EditorSettings:
    min_height: int = 3
    max_height: int = 15

    def __post_init__(self) -> None:
        if self.min_height < 1:
            raise ValueError("ui.editor.min_height must be at least 1")
        if self.max_height < self.min_height:
            raise ValueError(
                "ui.editor.max_height must be greater than or equal to min_height"
            )


@dataclass(frozen=True)
class KeyBindings:
    values: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: DEFAULT_KEY_BINDINGS
    )

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[str, ...]] = {}
        owners: dict[str, str] = {}
        for action, keys in self.values.items():
            if action not in DEFAULT_KEY_BINDINGS:
                raise ValueError(f"Unknown keybinding action: {action!r}")
            normalized_keys: list[str] = []
            for key in keys:
                value = key.strip().lower()
                if not value:
                    raise ValueError(f"Keybinding for {action!r} cannot be empty")
                previous = owners.get(value)
                if previous is not None and previous != action:
                    raise ValueError(
                        f"Keybinding {value!r} is assigned to both "
                        f"{previous!r} and {action!r}"
                    )
                owners[value] = action
                if value not in normalized_keys:
                    normalized_keys.append(value)
            normalized[action] = tuple(normalized_keys)
        for action, keys in DEFAULT_KEY_BINDINGS.items():
            normalized.setdefault(action, keys)
        object.__setattr__(self, "values", MappingProxyType(normalized))

    def keys(self, action: str) -> tuple[str, ...]:
        return self.values.get(action, ())

    def matches(self, action: str, key: str) -> bool:
        return key.lower() in self.keys(action)

    def primary(self, action: str) -> str | None:
        keys = self.keys(action)
        return keys[0] if keys else None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> KeyBindings:
        merged: dict[str, tuple[str, ...]] = dict(DEFAULT_KEY_BINDINGS)
        for action, raw_keys in values.items():
            if isinstance(raw_keys, str):
                keys = (raw_keys,)
            elif isinstance(raw_keys, Sequence) and not isinstance(raw_keys, bytes):
                if not all(isinstance(item, str) for item in raw_keys):
                    raise TypeError(
                        f"ui.keybindings.{action} must contain only strings"
                    )
                keys = tuple(raw_keys)
            else:
                raise TypeError(
                    f"ui.keybindings.{action} must be a string or list of strings"
                )
            merged[action] = keys
        return cls(merged)


@dataclass(frozen=True)
class VulcanoSettings:
    home: Path
    cwd: Path
    editor: EditorSettings = field(default_factory=EditorSettings)
    keybindings: KeyBindings = field(default_factory=KeyBindings)
    sources: tuple[Path, ...] = ()

    @classmethod
    def defaults(
        cls,
        *,
        cwd: str | Path | None = None,
        home: str | Path | None = None,
    ) -> VulcanoSettings:
        resolved_cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        resolved_home = (
            Path(home or os.environ.get("VULCANO_HOME", "~/.vulcano"))
            .expanduser()
            .resolve()
        )
        return cls(home=resolved_home, cwd=resolved_cwd)

    @classmethod
    def load(
        cls,
        *,
        cwd: str | Path | None = None,
        home: str | Path | None = None,
    ) -> VulcanoSettings:
        defaults = cls.defaults(cwd=cwd, home=home)
        paths = (
            defaults.home / "config.toml",
            defaults.cwd / ".vulcano" / "config.toml",
        )
        merged: dict[str, object] = {}
        loaded: list[Path] = []
        for path in paths:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                data = tomllib.load(stream)
            _merge_tables(merged, data)
            loaded.append(path)

        ui = _table(merged, "ui")
        editor = _table(ui, "editor")
        keybindings = _table(ui, "keybindings")
        return cls(
            home=defaults.home,
            cwd=defaults.cwd,
            editor=EditorSettings(
                min_height=_integer(editor, "min_height", 3),
                max_height=_integer(editor, "max_height", 15),
            ),
            keybindings=KeyBindings.from_mapping(keybindings),
            sources=tuple(loaded),
        )


def _merge_tables(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_tables(existing, value)
        elif isinstance(value, Mapping):
            nested: dict[str, object] = {}
            _merge_tables(nested, value)
            target[key] = nested
        else:
            target[key] = value


def _table(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = values.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a TOML table")
    return value


def _integer(values: Mapping[str, object], key: str, default: int) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"ui.editor.{key} must be an integer")
    return value
