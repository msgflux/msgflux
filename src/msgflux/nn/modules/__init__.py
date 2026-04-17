from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.modules.agent import Agent
    from msgflux.nn.modules.container import ModuleDict, ModuleList, Sequential
    from msgflux.nn.modules.embedder import Embedder
    from msgflux.nn.modules.generator import Generator
    from msgflux.nn.modules.mediamaker import MediaMaker
    from msgflux.nn.modules.module import Module
    from msgflux.nn.modules.predictor import Predictor
    from msgflux.nn.modules.searcher import Searcher
    from msgflux.nn.modules.speaker import Speaker
    from msgflux.nn.modules.tool import LocalTool, MCPTool, Tool, ToolLibrary
    from msgflux.nn.modules.transcriber import Transcriber

__all__ = [
    "Agent",
    "Embedder",
    "Generator",
    "LocalTool",
    "MCPTool",
    "MediaMaker",
    "Module",
    "ModuleDict",
    "ModuleList",
    "Predictor",
    "Searcher",
    "Sequential",
    "Speaker",
    "Tool",
    "ToolLibrary",
    "Transcriber",
]

_LAZY_IMPORTS = {
    "Agent": ("msgflux.nn.modules.agent", "Agent"),
    "Embedder": ("msgflux.nn.modules.embedder", "Embedder"),
    "Generator": ("msgflux.nn.modules.generator", "Generator"),
    "LocalTool": ("msgflux.nn.modules.tool", "LocalTool"),
    "MCPTool": ("msgflux.nn.modules.tool", "MCPTool"),
    "MediaMaker": ("msgflux.nn.modules.mediamaker", "MediaMaker"),
    "Module": ("msgflux.nn.modules.module", "Module"),
    "ModuleDict": ("msgflux.nn.modules.container", "ModuleDict"),
    "ModuleList": ("msgflux.nn.modules.container", "ModuleList"),
    "Predictor": ("msgflux.nn.modules.predictor", "Predictor"),
    "Searcher": ("msgflux.nn.modules.searcher", "Searcher"),
    "Sequential": ("msgflux.nn.modules.container", "Sequential"),
    "Speaker": ("msgflux.nn.modules.speaker", "Speaker"),
    "Tool": ("msgflux.nn.modules.tool", "Tool"),
    "ToolLibrary": ("msgflux.nn.modules.tool", "ToolLibrary"),
    "Transcriber": ("msgflux.nn.modules.transcriber", "Transcriber"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


# Please keep this list sorted
if __all__ != sorted(__all__):
    raise RuntimeError("__all__ must be sorted")
