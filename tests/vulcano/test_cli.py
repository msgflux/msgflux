from msgflux.vulcano.cli import _parser


def test_parser_accepts_transcript_view_override():
    args = _parser().parse_args(["--view", "compact"])

    assert args.view == "compact"


def test_parser_leaves_transcript_view_unset_by_default():
    args = _parser().parse_args([])

    assert args.view is None
