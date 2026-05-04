from argparse import Namespace

from msgflux.channels.social.slack import cli as slack_cli


def test_slack_auth_info_returns_bot_identity(monkeypatch):
    calls = []

    def fake_post(token, method, payload, timeout_s):
        calls.append((token, method, payload, timeout_s))
        return {
            "ok": True,
            "url": "https://example.slack.com/",
            "team": "msgflux",
            "team_id": "T123",
            "user": "msgflux-bot",
            "user_id": "U123",
            "bot_id": "B123",
        }

    monkeypatch.setattr(slack_cli, "_post_slack_api", fake_post)
    args = Namespace(
        slack_action="auth-info",
        bot_token="xoxb-token",
        bot_token_env="SLACK_BOT_TOKEN",
        timeout_s=3,
    )

    result = slack_cli._slack_auth_info(
        slack_cli._slack_bot_token(args), args.timeout_s
    )

    assert calls == [("xoxb-token", "auth.test", {}, 3)]
    assert result == {
        "ok": True,
        "url": "https://example.slack.com/",
        "team": "msgflux",
        "team_id": "T123",
        "user": "msgflux-bot",
        "user_id": "U123",
        "bot_id": "B123",
    }
