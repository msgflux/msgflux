from argparse import Namespace
from importlib import import_module

from msgflux.channels.http.app import create_app
from msgflux.channels.registry import load_registry_target


def run_server(args: Namespace) -> int:
    try:
        uvicorn = import_module("uvicorn")
    except ImportError as e:
        raise ImportError(
            "The msgflux server requires Uvicorn. Install it with "
            "`pip install msgflux[server]`."
        ) from e

    registry = load_registry_target(args.target)
    app = create_app(registry, title=args.title)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0
