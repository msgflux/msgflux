from pathlib import Path
from typing import Optional

from msgflux.auto.cache import AutoModuleCache
from msgflux.auto.exceptions import AutoModuleDownloadError
from msgflux.auto.sources.base import AutoModuleSource


class GitHubAutoModuleSource(AutoModuleSource):
    name = "github"
    raw_url_template = (
        "https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{path}"
    )

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
        parts = repo_id.split("/")
        if len(parts) != 2:
            raise ValueError(
                f"GitHub AutoModule repo id must be `owner/repo`: {repo_id}"
            )
        self.owner, self.repo = parts
        self.cache = AutoModuleCache(cache_dir)

    def download_file(self, filename: str, *, force_download: bool = False) -> Path:
        self.validate_filename(filename)
        module_path = self.cache.module_path(self.name, self.repo_id, self.revision)
        file_path = module_path / filename
        try:
            file_path.resolve().relative_to(module_path.resolve())
        except ValueError as exc:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "resolved cache path escapes the AutoModule cache directory.",
            ) from exc
        if file_path.exists() and not force_download:
            return file_path
        if self.local_files_only:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "`local_files_only=True` and file is not in cache.",
            )

        try:
            import httpx  # noqa: PLC0415
        except ImportError as exc:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "GitHub source requires `httpx`.",
            ) from exc

        file_path.parent.mkdir(parents=True, exist_ok=True)
        url = self.raw_url_template.format(
            owner=self.owner,
            repo=self.repo,
            revision=self.revision,
            path=filename,
        )
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                tmp_path = file_path.with_name(f"{file_path.name}.tmp")
                tmp_path.write_bytes(response.content)
                tmp_path.replace(file_path)
        except httpx.HTTPStatusError as exc:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                f"HTTP {exc.response.status_code} while downloading {url}.",
            ) from exc
        except httpx.RequestError as exc:
            raise AutoModuleDownloadError(self.repo_id, filename, str(exc)) from exc
        return file_path
