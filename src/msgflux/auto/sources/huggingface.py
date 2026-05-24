from pathlib import Path

from msgflux.auto.exceptions import AutoModuleDownloadError
from msgflux.auto.sources.base import AutoModuleSource


class HuggingFaceAutoModuleSource(AutoModuleSource):
    name = "huggingface"

    def download_file(self, filename: str, *, force_download: bool = False) -> Path:
        self.validate_filename(filename)
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
        except ImportError as exc:
            raise AutoModuleDownloadError(
                self.repo_id,
                filename,
                "Hugging Face source requires `huggingface_hub`.",
            ) from exc

        try:
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                cache_dir=self.cache_dir,
                force_download=force_download,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise AutoModuleDownloadError(self.repo_id, filename, str(exc)) from exc
        return Path(file_path)
