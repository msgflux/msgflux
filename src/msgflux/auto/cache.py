import os
import shutil
from pathlib import Path
from typing import Optional

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
        safe_repo_id = repo_id.replace("/", "--")
        return self.cache_dir / source / safe_repo_id / revision

    def clear(self) -> None:
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
