import os

from msgflux.channels.env import load_env_file


def test_load_env_file_reads_values_without_overriding(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TELEGRAM_BOT_TOKEN=file-token",
                "export TELEGRAM_WEBHOOK_SECRET='file-secret'",
                "EXISTING=from-file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("EXISTING", "from-env")

    loaded = load_env_file(env_file)

    assert loaded == 2
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "file-token"
    assert os.environ["TELEGRAM_WEBHOOK_SECRET"] == "file-secret"
    assert os.environ["EXISTING"] == "from-env"
