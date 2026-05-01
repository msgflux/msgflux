from msgflux.channels.cli import build_parser


def test_server_cli_parses_registry_target():
    parser = build_parser()

    args = parser.parse_args(
        [
            "server",
            "app.py:registry",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]
    )

    assert args.command == "server"
    assert args.target == "app.py:registry"
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.title is None
    assert args.description is None


def test_server_cli_default_port_avoids_common_dev_ports():
    parser = build_parser()

    args = parser.parse_args(["server", "app.py:registry"])

    assert args.port == 8010


def test_server_cli_parses_openapi_metadata_overrides():
    parser = build_parser()

    args = parser.parse_args(
        [
            "server",
            "app.py:registry",
            "--title",
            "Support Agents",
            "--description",
            "Support server",
        ]
    )

    assert args.title == "Support Agents"
    assert args.description == "Support server"
