"""Human service listings must not forward terminal controls from service data."""

import argparse
import asyncio
import json

from msgflux.runtime.service.local.listing import list_threads
from msgflux.runtime.service.records import ServiceThread

THREAD_ID = "ok\x1b[31mRED\x1b[0m\nnext"
AGENT_ID = "agent\rrow"
CWD = "/work/a\x1b]0;pwned\x07b\x7f"
UNICODE_THREAD_ID = "plain-café"


class _Client:
    async def active_runs(self):
        return ()

    async def threads(self):
        return (
            ServiceThread(
                thread_id=THREAD_ID,
                agent_id=AGENT_ID,
                cwd=CWD,
                interactive=True,
            ),
            ServiceThread(
                thread_id=UNICODE_THREAD_ID,
                agent_id="coding",
                cwd="/work/plain",
                interactive=False,
            ),
        )


def _list_threads(capsys, *, as_json):
    asyncio.run(list_threads(_Client(), argparse.Namespace(json=as_json, active=False)))
    return capsys.readouterr().out


def test_human_thread_table_escapes_ids_paths_and_row_breaks(capsys):
    output = _list_threads(capsys, as_json=False)

    assert all(control not in output for control in ("\x1b", "\x07", "\r", "\x7f"))
    assert len(output.splitlines()) == 3
    assert r"ok\x1b[31mRED\x1b[0m\nnext" in output
    assert r"agent\rrow" in output
    assert r"/work/a\x1b]0;pwned\x07b\x7f" in output
    assert UNICODE_THREAD_ID in output


def test_json_thread_listing_preserves_original_cells(capsys):
    output = _list_threads(capsys, as_json=True)

    assert len(output.splitlines()) == 1
    rows = json.loads(output)["threads"]
    assert [row["thread_id"] for row in rows] == [THREAD_ID, UNICODE_THREAD_ID]
    assert rows[0]["agent_id"] == AGENT_ID
    assert rows[0]["cwd"] == CWD
