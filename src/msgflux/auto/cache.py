import os
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from msgflux.envs import envs


def get_default_cache_dir() -> Path:
    if envs.auto_cache_dir:
        return Path(envs.auto_cache_dir).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "msgflux" / "auto"


class AutoModuleCache:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def module_path(self, source: str, repo_id: str, revision: str) -> Path:
        return (
            self.cache_dir
            / source
            / self._safe_component(repo_id)
            / self._safe_component(revision)
        )

    @staticmethod
    def _safe_component(value: str) -> str:
        digest = sha256(value.encode("utf-8")).hexdigest()[:12]
        quoted = quote(value, safe="")
        return f"{quoted}-{digest}"

    def clear(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
