"""Regression tests for documented MSGFLUX_* process environment settings."""

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

_PROBE = dedent(
    """
    import json

    from msgflux.envs import envs
    from msgflux.telemetry.config import msgtrace_settings

    print(
        "SETTINGS="
        + json.dumps(
            {
                "model_stop_after_attempt": envs.model_stop_after_attempt,
                "profile_cache_ttl": envs.profile_cache_ttl,
                "telemetry_capture_model_output": (
                    envs.telemetry_capture_model_output
                ),
                "telemetry_capture_state_dict": (
                    envs.telemetry_capture_state_dict
                ),
                "msgtrace_telemetry_enabled": (
                    msgtrace_settings.telemetry_enabled
                ),
            }
        )
    )
    """
)


def _read_settings(tmp_path, environment):
    """Read settings in a fresh process without ambient MSGFLUX_/MSGTRACE_ values."""
    process_env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("MSGFLUX_", "MSGTRACE_"))
    }
    process_env.update(environment)
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-c", _PROBE],
        cwd=tmp_path,
        env=process_env,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    payload = next(
        line
        for line in reversed(result.stdout.splitlines())
        if line.startswith("SETTINGS=")
    )
    return json.loads(payload.removeprefix("SETTINGS="))


_PROJECT_DOTENV_PROBE = dedent(
    """
    import json
    import os

    import msgflux as mf
    from msgflux.envs import envs
    from msgflux.telemetry.config import msgtrace_settings

    dotenv_keys = (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "MSGFLUX_CODEX_AUTH_FILE",
        "SSL_CERT_FILE",
        "MSGTRACE_OTLP_ENDPOINT",
        "MSGTRACE_TELEMETRY_ENABLED",
        "MSGFLUX_LOGGING_CONFIG_PATH",
        "MSGFLUX_MODEL_STOP_AFTER_ATTEMPT",
    )
    print(
        "SETTINGS="
        + json.dumps(
            {
                "dotenv_values": {key: os.environ.get(key) for key in dotenv_keys},
                "model_stop_after_attempt": envs.model_stop_after_attempt,
                "telemetry_enabled": msgtrace_settings.telemetry_enabled,
                "otlp_endpoint": msgtrace_settings.otlp_endpoint,
                "load_dotenv_available": callable(mf.load_dotenv),
            }
        )
    )
    """
)


def _read_project_settings(tmp_path, probe, *, explicit_dotenv=False, environment=None):
    """Run a settings probe with a minimal child environment and isolated source."""
    process_env = {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(_SOURCE_ROOT),
    }
    process_env.update(environment or {})
    if explicit_dotenv:
        probe = "import msgflux as mf\nmf.load_dotenv()\n" + probe
    result = subprocess.run(  # noqa: S603 - fixed interpreter and test script
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=process_env,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    payload = next(
        line
        for line in reversed(result.stdout.splitlines())
        if line.startswith("SETTINGS=")
    )
    return json.loads(payload.removeprefix("SETTINGS="))


def _write_project_dotenv(tmp_path):
    logging_path = tmp_path / "logging.json"
    logging_path.write_text('{"version": 1}', encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=FAKE_ACCOUNT_KEY_DO_NOT_USE",
                "OPENAI_BASE_URL=https://attacker.example.test/v1",
                "HTTP_PROXY=http://proxy.example.test:8080",
                "HTTPS_PROXY=http://proxy.example.test:8080",
                "ALL_PROXY=http://proxy.example.test:8080",
                "MSGFLUX_CODEX_AUTH_FILE=/tmp/fake-codex-auth.json",
                "SSL_CERT_FILE=/tmp/fake-ca.pem",
                "MSGTRACE_OTLP_ENDPOINT=https://telemetry.example.test/v1",
                "MSGTRACE_TELEMETRY_ENABLED=true",
                f"MSGFLUX_LOGGING_CONFIG_PATH={logging_path}",
                "MSGFLUX_MODEL_STOP_AFTER_ATTEMPT=17",
            )
        ),
        encoding="utf-8",
    )


def test_project_dotenv_is_not_loaded_on_import(tmp_path):
    _write_project_dotenv(tmp_path)

    settings = _read_project_settings(tmp_path, _PROJECT_DOTENV_PROBE)

    assert settings["dotenv_values"] == {
        "OPENAI_API_KEY": None,
        "OPENAI_BASE_URL": None,
        "HTTP_PROXY": None,
        "HTTPS_PROXY": None,
        "ALL_PROXY": None,
        "MSGFLUX_CODEX_AUTH_FILE": None,
        "SSL_CERT_FILE": None,
        "MSGTRACE_OTLP_ENDPOINT": None,
        "MSGTRACE_TELEMETRY_ENABLED": None,
        "MSGFLUX_LOGGING_CONFIG_PATH": None,
        "MSGFLUX_MODEL_STOP_AFTER_ATTEMPT": None,
    }
    assert settings["model_stop_after_attempt"] == 5
    assert settings["telemetry_enabled"] is False
    assert settings["otlp_endpoint"] == "http://localhost:4318"
    assert settings["load_dotenv_available"] is True


def test_explicit_load_dotenv_reads_project_file(tmp_path):
    _write_project_dotenv(tmp_path)

    settings = _read_project_settings(
        tmp_path, _PROJECT_DOTENV_PROBE, explicit_dotenv=True
    )

    assert settings["dotenv_values"]["OPENAI_API_KEY"] == "FAKE_ACCOUNT_KEY_DO_NOT_USE"
    assert settings["dotenv_values"]["OPENAI_BASE_URL"] == (
        "https://attacker.example.test/v1"
    )
    assert settings["model_stop_after_attempt"] == 17
    assert settings["telemetry_enabled"] is True
    assert settings["otlp_endpoint"] == "https://telemetry.example.test/v1"


def test_explicit_load_dotenv_preserves_exported_values(tmp_path):
    _write_project_dotenv(tmp_path)

    settings = _read_project_settings(
        tmp_path,
        _PROJECT_DOTENV_PROBE,
        explicit_dotenv=True,
        environment={
            "OPENAI_API_KEY": "FAKE_EXPORTED_KEY_DO_NOT_USE",
            "MSGFLUX_MODEL_STOP_AFTER_ATTEMPT": "9",
            "MSGTRACE_TELEMETRY_ENABLED": "false",
            "MSGTRACE_OTLP_ENDPOINT": "https://exported.example.test/v1",
        },
    )

    assert settings["dotenv_values"]["OPENAI_API_KEY"] == (
        "FAKE_EXPORTED_KEY_DO_NOT_USE"
    )
    assert settings["model_stop_after_attempt"] == 9
    assert settings["telemetry_enabled"] is False
    assert settings["otlp_endpoint"] == "https://exported.example.test/v1"


def test_uppercase_msgflux_prefix_applies_numeric_settings(tmp_path):
    settings = _read_settings(
        tmp_path,
        {
            "MSGFLUX_MODEL_STOP_AFTER_ATTEMPT": "2",
            "MSGFLUX_PROFILE_CACHE_TTL": "123",
        },
    )

    assert settings["model_stop_after_attempt"] == 2
    assert settings["profile_cache_ttl"] == 123


def test_uppercase_msgflux_prefix_applies_telemetry_opt_outs(tmp_path):
    settings = _read_settings(
        tmp_path,
        {
            "MSGFLUX_TELEMETRY_CAPTURE_MODEL_OUTPUT": "false",
            "MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT": "true",
            # MsgTraceSettings already documents and tests this uppercase pattern.
            "MSGTRACE_TELEMETRY_ENABLED": "true",
        },
    )

    assert settings["telemetry_capture_model_output"] is False
    assert settings["telemetry_capture_state_dict"] is True
    assert settings["msgtrace_telemetry_enabled"] is True
