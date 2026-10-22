"""Configuration for msgtrace-sdk integration.

This module provides a separate configuration for msgtrace-sdk
environment variables, keeping msgflux and msgtrace configurations
separated but integrated.
"""

import os
from typing import Literal, Optional

from msgspec_ext import BaseSettings, SettingsConfigDict


class MsgTraceSettings(BaseSettings):
    """Settings for msgtrace-sdk configuration.

    These settings control the msgtrace-sdk behavior and are
    mapped to MSGTRACE_* environment variables.
    """

    model_config = SettingsConfigDict(
        # Project .env files are opt-in via msgflux.load_dotenv().
        env_prefix="MSGTRACE_",
    )

    # Enable telemetry tracing
    telemetry_enabled: bool = False

    # Span exporter type: "console", "otlp", or other supported types
    exporter: str = "console"

    # OTLP endpoint for trace export
    otlp_endpoint: str = "http://localhost:4318"

    # Service name for telemetry
    service_name: str = "msgflux"

    # Sampling ratio for traces (None for default, or string like "0.5")
    sampling_ratio: Optional[str] = None

    # Capture platform information (CPU, OS, Python version)
    capture_platform: bool = True

    # Maximum retries for telemetry operations
    max_retries: int = 3


# Global instance
msgtrace_settings = MsgTraceSettings()


def configure_msgtrace(
    *,
    enabled: Optional[bool] = None,
    exporter: Optional[Literal["console", "otlp"]] = None,
    otlp_endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    sampling_ratio: Optional[str] = None,
    capture_platform: Optional[bool] = None,
    max_retries: Optional[int] = None,
) -> None:
    """Configure msgtrace-sdk settings.

    This function allows programmatic configuration of msgtrace-sdk
    without directly setting environment variables.

    Args:
        enabled: Enable telemetry tracking
        exporter: Exporter type ("otlp" or "console")
        otlp_endpoint: OTLP endpoint URL
        service_name: Service name for telemetry
        sampling_ratio: Sampling ratio for traces
        capture_platform: Capture platform details
        max_retries: Maximum retries for telemetry operations
    """
    if enabled is not None:
        msgtrace_settings.telemetry_enabled = enabled
        os.environ["MSGTRACE_TELEMETRY_ENABLED"] = "true" if enabled else "false"

    if exporter is not None:
        msgtrace_settings.exporter = exporter
        os.environ["MSGTRACE_EXPORTER"] = exporter

    if otlp_endpoint is not None:
        msgtrace_settings.otlp_endpoint = otlp_endpoint
        os.environ["MSGTRACE_OTLP_ENDPOINT"] = otlp_endpoint

    if service_name is not None:
        msgtrace_settings.service_name = service_name
        os.environ["MSGTRACE_SERVICE_NAME"] = service_name

    if sampling_ratio is not None:
        msgtrace_settings.sampling_ratio = sampling_ratio
        os.environ["MSGTRACE_SAMPLING_RATIO"] = sampling_ratio

    if capture_platform is not None:
        msgtrace_settings.capture_platform = capture_platform
        os.environ["MSGTRACE_CAPTURE_PLATFORM"] = (
            "true" if capture_platform else "false"
        )

    if max_retries is not None:
        msgtrace_settings.max_retries = max_retries
        os.environ["MSGTRACE_MAX_RETRIES"] = str(max_retries)
