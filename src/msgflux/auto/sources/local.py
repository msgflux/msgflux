from pathlib import Path
from typing import Optional

from msgflux.auto.exceptions import AutoModuleDownloadError
from msgflux.auto.sources.base import AutoModuleSource


class LocalAutoModuleSource(AutoModuleSource):
    name = "local"

    def __init__(
        self,
        repo_id: str,
        *,
        revision: Optional[str],
        cache_dir: Path,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(
            repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.root = Path(repo_id).expanduser()

    def download_file(self, filename: str, *, force_download: bool = False) -> Path:
        del force_download
        self.validate_filename(filename)
        root = self.root.resolve()
        file_path = (self.root / filename).resolve()
        try:
            file_path.relative_to(root)
        except ValueError as exc:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "resolved local path escapes the AutoModule directory.",
            ) from exc
        if not file_path.exists():
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "file does not exist.",
            )
        return file_path
