from argparse import Namespace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from msgflux.channels.http.app import create_app
from msgflux.channels.registry import load_registry_target
from msgflux.logger import logger

_REMOTE_TIMEOUT_SECONDS = 15
_REMOTE_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB
_CACHE_DIR = Path.home() / ".cache" / "msgflux" / "servers"


def run_server(args: Namespace) -> int:
    try:
        uvicorn = import_module("uvicorn")
    except ImportError as e:
        raise ImportError(
            "The msgflux server requires Uvicorn. Install it with "
            "`pip install msgflux[server]`."
        ) from e

    target = _resolve_server_target(
        args.target,
        trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
    )
    registry = load_registry_target(target)
    app = create_app(registry, title=args.title)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


def _resolve_server_target(target: str, *, trust_remote_code: bool) -> str:
    remote_url, attr_name = _split_remote_target(target)
    if remote_url is None:
        return target

    if not trust_remote_code:
        raise ValueError(
            "Refusing to execute remote code without --trust-remote-code. "
            "Pass this flag to allow downloading and running the remote server file."
        )

    downloaded = _download_remote_target(remote_url)
    if attr_name:
        return f"{downloaded}:{attr_name}"
    return str(downloaded)


def _is_http_url(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _split_remote_target(target: str) -> tuple[str | None, str | None]:
    marker = ".py:"
    idx = target.rfind(marker)
    if idx != -1:
        url_part = target[: idx + 3]
        attr_part = target[idx + len(marker) :]
        if _is_http_url(url_part) and attr_part:
            return url_part, attr_part

    if _is_http_url(target):
        return target, None

    return None, None


def _download_remote_target(url: str) -> Path:
    logger.info("Downloading server file from %s", url)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{sha256(url.encode('utf-8')).hexdigest()[:16]}.py"
    destination = _CACHE_DIR / filename

    # URL scheme is validated by `_split_remote_target`/`_is_http_url` before download.
    with urlopen(url, timeout=_REMOTE_TIMEOUT_SECONDS) as response:  # noqa: S310
        status = getattr(response, "status", None)
        if status and status >= 400:
            raise RuntimeError(f"Failed to download `{url}`: HTTP {status}")

        data = response.read(_REMOTE_MAX_BYTES + 1)
        if len(data) > _REMOTE_MAX_BYTES:
            raise RuntimeError(
                f"Remote target exceeds max size of {_REMOTE_MAX_BYTES} bytes: `{url}`"
            )

    destination.write_bytes(data)
    logger.info("Downloaded server file to %s", destination)
    return destination
