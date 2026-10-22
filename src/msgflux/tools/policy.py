"""Immutable, invocation-local tool authorization policy."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

import msgspec


def normalize_tool_filter_values(values: Any, *, key: str) -> frozenset[str]:
    if isinstance(values, str):
        if not values:
            raise ValueError(
                f"`tool_filter['{key}']` must be a non-empty string or list of strings"
            )
        return frozenset((values,))
    if isinstance(values, list):
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"`tool_filter['{key}']` must contain only non-empty strings"
            )
        return frozenset(values)
    raise ValueError(
        f"`tool_filter['{key}']` must be a string or list of strings, "
        f"given `{type(values)}`"
    )


def normalize_tool_filter(tool_filter: Any) -> tuple[str, frozenset[str]]:
    if not isinstance(tool_filter, dict):
        raise ValueError(f"`tool_filter` must be a dict, given `{type(tool_filter)}`")
    keys = set(tool_filter)
    valid_keys = {"allow", "block"}
    if not keys:
        raise ValueError("`tool_filter` must contain 'allow' or 'block' key")
    if keys - valid_keys:
        raise ValueError(
            f"`tool_filter` contains invalid keys: {keys - valid_keys}. "
            f"Valid keys are: {valid_keys}"
        )
    if len(keys) > 1:
        raise ValueError(
            "`tool_filter` must contain only one key: 'allow' or 'block', "
            f"got both: {keys}"
        )
    mode = next(iter(keys))
    return mode, normalize_tool_filter_values(tool_filter[mode], key=mode)


class ToolFilterPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Snapshot of the logical tool names authorized for one Agent invocation."""

    allowed_names: frozenset[str]

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_names

    def to_state(self) -> dict[str, list[str]]:
        return {"allowed_names": sorted(self.allowed_names)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ToolFilterPolicy:
        if not isinstance(state, Mapping):
            raise ValueError("Invalid checkpointed tool-filter policy")
        names = state.get("allowed_names")
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("Invalid checkpointed tool-filter policy")
        return cls(allowed_names=frozenset(names))


_POLICIES: ContextVar[dict[int, ToolFilterPolicy | None] | None] = ContextVar(
    "msgflux_tool_filter_policies", default=None
)


def get_tool_filter_policy(library: object) -> ToolFilterPolicy | None:
    return (_POLICIES.get() or {}).get(id(library))


@contextmanager
def bind_tool_filter_policy(
    library: object,
    policy: ToolFilterPolicy | None,
) -> Iterator[None]:
    policies = dict(_POLICIES.get() or {})
    policies[id(library)] = policy
    token = _POLICIES.set(policies)
    try:
        yield
    finally:
        _POLICIES.reset(token)


__all__ = [
    "ToolFilterPolicy",
    "bind_tool_filter_policy",
    "get_tool_filter_policy",
    "normalize_tool_filter",
    "normalize_tool_filter_values",
]
