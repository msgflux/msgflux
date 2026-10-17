"""Tests for msgflux.telemetry.config module."""

import os

import msgspec
import pytest

from msgflux.telemetry.config import (
    MsgTraceSettings,
    configure_msgtrace,
    msgtrace_settings,
)


@pytest.fixture(autouse=True)
def restore_msgtrace_configuration():
    """Keep programmatic configuration from leaking into other tests or children."""
    settings = msgspec.structs.asdict(msgtrace_settings)
    environment = {
        f"MSGTRACE_{name.upper()}": os.environ.get(f"MSGTRACE_{name.upper()}")
        for name in settings
    }
    yield
    for name, value in environment.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    for name, value in settings.items():
        setattr(msgtrace_settings, name, value)


class TestMsgTraceSettings:
    """Test suite for MsgTraceSettings configuration class."""

    def test_settings_initialization_defaults(self):
        """Test MsgTraceSettings initialization with default values."""
        settings = MsgTraceSettings()

        assert settings.telemetry_enabled is False
        assert settings.exporter == "console"
        assert settings.otlp_endpoint == "http://localhost:4318"
        assert settings.service_name == "msgflux"
        assert settings.capture_platform is True
        assert settings.max_retries == 3

    def test_settings_from_env_variables(self, monkeypatch):
        """Test MsgTraceSettings loads from environment variables."""
        monkeypatch.setenv("MSGTRACE_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("MSGTRACE_EXPORTER", "otlp")
        monkeypatch.setenv("MSGTRACE_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("MSGTRACE_SERVICE_NAME", "test-service")

        settings = MsgTraceSettings()

        assert settings.telemetry_enabled is True
        assert settings.exporter == "otlp"
        assert settings.otlp_endpoint == "http://localhost:4317"
        assert settings.service_name == "test-service"

    def test_settings_telemetry_enabled_false(self, monkeypatch):
        """Test telemetry_enabled with false value."""
        monkeypatch.setenv("MSGTRACE_TELEMETRY_ENABLED", "false")

        settings = MsgTraceSettings()

        assert settings.telemetry_enabled is False

    def test_global_settings_instance(self):
        """Test that global msgtrace_settings instance exists."""
        assert msgtrace_settings is not None
        assert hasattr(msgtrace_settings, "telemetry_enabled")
        assert hasattr(msgtrace_settings, "exporter")


class TestConfigureMsgtrace:
    """Test suite for configure_msgtrace function."""

    def test_configure_enabled(self):
        """Test configuring enabled parameter."""
        configure_msgtrace(enabled=True)

        assert os.environ.get("MSGTRACE_TELEMETRY_ENABLED") == "true"
        assert msgtrace_settings.telemetry_enabled is True

    def test_configure_enabled_false(self):
        """Test configuring enabled=False parameter."""
        configure_msgtrace(enabled=False)

        assert os.environ.get("MSGTRACE_TELEMETRY_ENABLED") == "false"
        assert msgtrace_settings.telemetry_enabled is False

    def test_configure_exporter(self):
        """Test configuring exporter parameter."""
        configure_msgtrace(exporter="otlp")

        assert os.environ.get("MSGTRACE_EXPORTER") == "otlp"
        assert msgtrace_settings.exporter == "otlp"

    def test_configure_otlp_endpoint(self):
        """Test configuring otlp_endpoint parameter."""
        configure_msgtrace(otlp_endpoint="http://localhost:4317")

        assert os.environ.get("MSGTRACE_OTLP_ENDPOINT") == "http://localhost:4317"
        assert msgtrace_settings.otlp_endpoint == "http://localhost:4317"

    def test_configure_service_name(self):
        """Test configuring service_name parameter."""
        configure_msgtrace(service_name="my-service")

        assert os.environ.get("MSGTRACE_SERVICE_NAME") == "my-service"
        assert msgtrace_settings.service_name == "my-service"

    def test_configure_sampling_ratio(self):
        """Test configuring sampling_ratio parameter."""
        configure_msgtrace(sampling_ratio="0.5")

        assert os.environ.get("MSGTRACE_SAMPLING_RATIO") == "0.5"
        assert msgtrace_settings.sampling_ratio == "0.5"

    def test_configure_capture_platform(self):
        """Test configuring capture_platform parameter."""
        configure_msgtrace(capture_platform=False)

        assert os.environ.get("MSGTRACE_CAPTURE_PLATFORM") == "false"
        assert msgtrace_settings.capture_platform is False

    def test_configure_max_retries(self):
        """Test configuring max_retries parameter."""
        configure_msgtrace(max_retries=5)

        assert os.environ.get("MSGTRACE_MAX_RETRIES") == "5"
        assert msgtrace_settings.max_retries == 5

    def test_configure_multiple_parameters(self):
        """Test configuring multiple parameters at once."""
        configure_msgtrace(
            enabled=True,
            exporter="otlp",
            otlp_endpoint="http://localhost:9999",
            service_name="multi-test",
        )

        assert msgtrace_settings.telemetry_enabled is True
        assert msgtrace_settings.exporter == "otlp"
        assert msgtrace_settings.otlp_endpoint == "http://localhost:9999"
        assert msgtrace_settings.service_name == "multi-test"
