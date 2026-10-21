"""Regression tests for documented MSGFLUX_* process environment settings."""

import json
import os
import subprocess
import sys
from textwrap import dedent

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
