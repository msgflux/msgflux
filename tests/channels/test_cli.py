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
