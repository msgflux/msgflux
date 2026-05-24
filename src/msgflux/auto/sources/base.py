from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class AutoModuleSource(ABC):
    name = "base"

    def __init__(
        self,
        repo_id: str,
        *,
        revision: Optional[str],
        cache_dir: Path,
        local_files_only: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision or "main"
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only

    @abstractmethod
    def download_file(self, filename: str, *, force_download: bool = False) -> Path:
        """Return a local path for a file in the remote module."""

    @staticmethod
    def validate_filename(filename: str) -> None:
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"AutoModule file paths must be relative and stay inside the "
                f"repository, got `{filename}`."
            )
