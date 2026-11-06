import sys

import pytest

from msgflux._private import memory


@pytest.fixture
def fake_trim(monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "_resolved", True)
    monkeypatch.setattr(memory, "_malloc_trim", calls.append)
    monkeypatch.setattr(memory, "_last_trim", float("-inf"))
    return calls


def test_release_is_throttled_process_wide(fake_trim, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(memory.time, "monotonic", lambda: clock[0])

    assert memory.release_freed_memory(background=False) is True
    assert memory.release_freed_memory(background=False) is False
    clock[0] += memory.TRIM_INTERVAL_SECONDS
    assert memory.release_freed_memory(background=False) is True
    assert fake_trim == [0, 0]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc only")
def test_env_switch_disables_trim(monkeypatch):
    monkeypatch.setenv("MSGFLUX_MALLOC_TRIM", "0")
    monkeypatch.setattr(memory, "_resolved", False)
    monkeypatch.setattr(memory, "_malloc_trim", None)
    monkeypatch.setattr(memory, "_last_trim", float("-inf"))

    assert memory.release_freed_memory(background=False) is False
