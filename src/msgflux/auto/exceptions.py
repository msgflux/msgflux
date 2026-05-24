class AutoModuleError(Exception):
    """Base exception for AutoModule failures."""


class AutoModuleConfigurationError(AutoModuleError):
    """Raised when a remote module manifest is invalid."""

    def __init__(self, repo_id: str, message: str):
        super().__init__(f"Invalid AutoModule `{repo_id}`: {message}")


class AutoModuleDownloadError(AutoModuleError):
    """Raised when an AutoModule file cannot be downloaded."""

    def __init__(self, repo_id: str, filename: str, message: str):
        super().__init__(
            f"Could not download `{filename}` from AutoModule `{repo_id}`: {message}"
        )


class AutoModuleSecurityError(AutoModuleError):
    """Raised when loading remote Python code without explicit trust."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"AutoModule `{repo_id}` requires `trust_remote_code=True` because "
            "loading it executes remote Python code."
        )
