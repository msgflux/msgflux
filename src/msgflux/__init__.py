from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgspec_ext import load_dotenv

    from msgflux.auto import AutoModule
    from msgflux.cache import response_cache
    from msgflux.core.dotdict import dotdict
    from msgflux.core.examples import Example
    from msgflux.core.message import Message
    from msgflux.core.registry import Registry
    from msgflux.data.dbs import DB
    from msgflux.data.parsers import Parser
    from msgflux.data.retrievers import Retriever
    from msgflux.data.types import Audio, File, Image, Video
    from msgflux.dsl.inline import Inline
    from msgflux.dsl.signature import InputField, OutputField, Signature
    from msgflux.envs import set_envs
    from msgflux.exceptions import TaskError
    from msgflux.models import Model
    from msgflux.models.gateway import ModelGateway
    from msgflux.runtime import (
        AgentSkill,
        AgentSkillManager,
        SkillsConfig,
        default_skill_paths,
        parse_skill_file,
    )
    from msgflux.telemetry import Spans
    from msgflux.tools.config import tool_config
    from msgflux.utils.chat import ChatBlock, ChatML
    from msgflux.utils.console import cprint
    from msgflux.utils.inspect import get_fn_name
    from msgflux.utils.msgspec import load, msgspec_dumps, save

__all__ = [
    "AutoModule",
    "DB",
    "Audio",
    "AgentSkill",
    "AgentSkillManager",
    "ChatBlock",
    "ChatML",
    "Example",
    "File",
    "Image",
    "Inline",
    "InputField",
    "Message",
    "Model",
    "ModelGateway",
    "OutputField",
    "Parser",
    "Registry",
    "Retriever",
    "Signature",
    "Spans",
    "SkillsConfig",
    "TaskError",
    "Video",
    "cprint",
    "default_skill_paths",
    "dotdict",
    "get_fn_name",
    "load",
    "load_dotenv",
    "msgspec_dumps",
    "parse_skill_file",
    "response_cache",
    "save",
    "set_envs",
    "tool_config",
]

_LAZY_IMPORTS = {
    "AutoModule": ("msgflux.auto", "AutoModule"),
    "Audio": ("msgflux.data.types", "Audio"),
    "AgentSkill": ("msgflux.runtime", "AgentSkill"),
    "AgentSkillManager": ("msgflux.runtime", "AgentSkillManager"),
    "ChatBlock": ("msgflux.utils.chat", "ChatBlock"),
    "ChatML": ("msgflux.utils.chat", "ChatML"),
    "DB": ("msgflux.data.dbs", "DB"),
    "Example": ("msgflux.core.examples", "Example"),
    "File": ("msgflux.data.types", "File"),
    "Image": ("msgflux.data.types", "Image"),
    "Inline": ("msgflux.dsl.inline", "Inline"),
    "InputField": ("msgflux.dsl.signature", "InputField"),
    "Message": ("msgflux.core.message", "Message"),
    "Model": ("msgflux.models", "Model"),
    "ModelGateway": ("msgflux.models.gateway", "ModelGateway"),
    "OutputField": ("msgflux.dsl.signature", "OutputField"),
    "Parser": ("msgflux.data.parsers", "Parser"),
    "Registry": ("msgflux.core.registry", "Registry"),
    "Retriever": ("msgflux.data.retrievers", "Retriever"),
    "Signature": ("msgflux.dsl.signature", "Signature"),
    "SkillsConfig": ("msgflux.runtime", "SkillsConfig"),
    "Spans": ("msgflux.telemetry", "Spans"),
    "TaskError": ("msgflux.exceptions", "TaskError"),
    "Video": ("msgflux.data.types", "Video"),
    "cprint": ("msgflux.utils.console", "cprint"),
    "default_skill_paths": ("msgflux.runtime", "default_skill_paths"),
    "dotdict": ("msgflux.core.dotdict", "dotdict"),
    "get_fn_name": ("msgflux.utils.inspect", "get_fn_name"),
    "load": ("msgflux.utils.msgspec", "load"),
    "load_dotenv": ("msgspec_ext", "load_dotenv"),
    "msgspec_dumps": ("msgflux.utils.msgspec", "msgspec_dumps"),
    "parse_skill_file": ("msgflux.runtime", "parse_skill_file"),
    "response_cache": ("msgflux.cache", "response_cache"),
    "save": ("msgflux.utils.msgspec", "save"),
    "set_envs": ("msgflux.envs", "set_envs"),
    "tool_config": ("msgflux.tools.config", "tool_config"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
