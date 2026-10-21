"""Staging directories are cleaned if their descriptor cannot be opened."""

import errno

import pytest

from msgflux.runtime import LocalToolResultStore, tool_results


def _fail_staging_open(monkeypatch, error):
    original_open = tool_results.os.open

    def open_or_fail(path, flags, *args, **kwargs):
        if isinstance(path, str) and path.startswith(".pending-"):
            raise error
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tool_results.os, "open", open_or_fail)


def test_staging_open_failure_preserves_error_and_removes_empty_directory(
    tmp_path, monkeypatch
):
    store = LocalToolResultStore(tmp_path)
    error = OSError(errno.EMFILE, "descriptor exhaustion")
    _fail_staging_open(monkeypatch, error)

    with pytest.raises(OSError) as caught:
        store.put([b"result"])

    assert caught.value is error
    assert store.usage().pending == 0
    assert list(tmp_path.iterdir()) == []


def test_staging_open_cleanup_failure_is_noted_on_original_error(tmp_path, monkeypatch):
    store = LocalToolResultStore(tmp_path)
    error = OSError(errno.EMFILE, "descriptor exhaustion")
    with monkeypatch.context() as patch:
        _fail_staging_open(patch, error)

        def fail_rmdir(*_args, **_kwargs):
            raise PermissionError("cleanup denied")

        patch.setattr(tool_results.os, "rmdir", fail_rmdir)
        with pytest.raises(OSError) as caught:
            store.put([b"result"])

    assert caught.value is error
    cleanup_notes = [
        note
        for note in error.__notes__
        if "Unable to remove unopened staging directory" in note
    ]
    assert len(cleanup_notes) == 1
    assert "cleanup denied" in cleanup_notes[0]
    assert store.usage().pending == 1


def test_staging_open_success_still_publishes_result(tmp_path):
    store = LocalToolResultStore(tmp_path)

    reference = store.put([b"result"])

    assert store.read(reference) == b"result"
    usage = store.usage()
    assert usage.results == 1
    assert usage.pending == 0
