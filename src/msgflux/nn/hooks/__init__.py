from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.hooks.events import (
        AfterTool,
        AgentContext,
        BeforeCompaction,
        BeforeResume,
        BeforeRun,
        BeforeTool,
        BeforeToolDispatch,
        ConversationContext,
        ModelContext,
        ModelRequestContext,
        ModelResponseContext,
        NotificationContext,
        OutputContext,
        RunEndContext,
        ToolCatalogContext,
        ToolFeedbackContext,
    )
    from msgflux.nn.hooks.guard import Guard
    from msgflux.nn.hooks.hook import Hook, RemovableHandle

__all__ = [
    "AgentContext",
    "AfterTool",
    "BeforeResume",
    "BeforeCompaction",
    "BeforeRun",
    "BeforeTool",
    "BeforeToolDispatch",
    "ConversationContext",
    "Guard",
    "Hook",
    "OutputContext",
    "RemovableHandle",
    "ModelContext",
    "ModelRequestContext",
    "ModelResponseContext",
    "NotificationContext",
    "RunEndContext",
    "ToolCatalogContext",
    "ToolFeedbackContext",
    "ContinuationContext",
]

_LAZY_IMPORTS = {
    "ContinuationContext": ("msgflux.nn.hooks.events", "ContinuationContext"),
    "AgentContext": ("msgflux.nn.hooks.events", "AgentContext"),
    "AfterTool": ("msgflux.nn.hooks.events", "AfterTool"),
    "BeforeResume": ("msgflux.nn.hooks.events", "BeforeResume"),
    "BeforeCompaction": ("msgflux.nn.hooks.events", "BeforeCompaction"),
    "BeforeRun": ("msgflux.nn.hooks.events", "BeforeRun"),
    "BeforeTool": ("msgflux.nn.hooks.events", "BeforeTool"),
    "BeforeToolDispatch": ("msgflux.nn.hooks.events", "BeforeToolDispatch"),
    "ConversationContext": ("msgflux.nn.hooks.events", "ConversationContext"),
    "Guard": ("msgflux.nn.hooks.guard", "Guard"),
    "Hook": ("msgflux.nn.hooks.hook", "Hook"),
    "OutputContext": ("msgflux.nn.hooks.events", "OutputContext"),
    "RemovableHandle": ("msgflux.nn.hooks.hook", "RemovableHandle"),
    "ModelContext": ("msgflux.nn.hooks.events", "ModelContext"),
    "ModelRequestContext": ("msgflux.nn.hooks.events", "ModelRequestContext"),
    "ModelResponseContext": ("msgflux.nn.hooks.events", "ModelResponseContext"),
    "NotificationContext": ("msgflux.nn.hooks.events", "NotificationContext"),
    "RunEndContext": ("msgflux.nn.hooks.events", "RunEndContext"),
    "ToolCatalogContext": ("msgflux.nn.hooks.events", "ToolCatalogContext"),
    "ToolFeedbackContext": ("msgflux.nn.hooks.events", "ToolFeedbackContext"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
