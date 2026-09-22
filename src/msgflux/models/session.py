"""Session/cache header helpers for gateway providers.

Gateway providers (xAI, OpenCode) use sticky routing and prompt caching
keyed by a client-supplied session identifier. The active thread id from
the execution context is the stable per-conversation value for it.
"""

from typing import Any, Dict

from msgflux.runtime.context import get_thread_id

USER_AGENT = "msgflux"


def session_extra_headers(session_header: str) -> Dict[str, str]:
    """Build request headers with client identity and active-thread session.

    Always identifies as `msgflux`; adds `session_header` only when a
    thread id is active so anonymous calls stay header-clean.
    """
    headers = {"User-Agent": USER_AGENT}
    thread_id = get_thread_id()
    if thread_id:
        headers[session_header] = thread_id
    return headers


def merge_session_headers(
    params: Dict[str, Any], session_header: str
) -> Dict[str, Any]:
    """Merge session headers into request params without dropping caller's."""
    extra_headers = dict(params.get("extra_headers") or {})
    for key, value in session_extra_headers(session_header).items():
        extra_headers.setdefault(key, value)
    params["extra_headers"] = extra_headers
    return params
