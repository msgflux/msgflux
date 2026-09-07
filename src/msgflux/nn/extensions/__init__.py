"""Composable Agent and ToolLibrary extensions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.extensions.base import AgentExtension, AgentExtensionHandle
    from msgflux.nn.extensions.artifact import (
        Artifact,
        ArtifactExtension,
        ArtifactReferenceRenderer,
        ArtifactRegistry,
    )
    from msgflux.nn.extensions.compaction import (
        CONTEXT_COMPACTION_CAPABILITY,
        CompactionExtension,
        CompactionPolicy,
    )
    from msgflux.nn.extensions.feedback import DefaultToolFeedbackExtension
    from msgflux.nn.extensions.prompt import (
        CurrentDateExtension,
        FewShotExamplesExtension,
        ToolUsageGuidanceExtension,
    )
    from msgflux.nn.extensions.skills import SkillsExtension
    from msgflux.nn.extensions.tool_library import (
        BackgroundTasksExtension,
        MCPServersExtension,
        ToolLibraryExtension,
        ToolLibraryExtensionHandle,
        ToolSearchExtension,
    )
    from msgflux.nn.modules.tool_runtime import (
        ToolContextProvider,
        ToolDispatch,
        ToolPolicy,
    )

__all__ = [
    "AgentExtension",
    "AgentExtensionHandle",
    "Artifact",
    "ArtifactExtension",
    "ArtifactReferenceRenderer",
    "ArtifactRegistry",
    "CompactionExtension",
    "CompactionPolicy",
    "CONTEXT_COMPACTION_CAPABILITY",
    "BackgroundTasksExtension",
    "CurrentDateExtension",
    "FewShotExamplesExtension",
    "SkillsExtension",
    "MCPServersExtension",
    "ToolLibraryExtension",
    "ToolLibraryExtensionHandle",
    "ToolDispatch",
    "ToolContextProvider",
    "ToolPolicy",
    "ToolSearchExtension",
    "ToolUsageGuidanceExtension",
    "DefaultToolFeedbackExtension",
    "ToolTurnLimitExtension",
]

_LAZY_IMPORTS = {
    "ToolTurnLimitExtension": ("msgflux.nn.extensions.limits", "ToolTurnLimitExtension"),
    "AgentExtension": ("msgflux.nn.extensions.base", "AgentExtension"),
    "AgentExtensionHandle": (
        "msgflux.nn.extensions.base",
        "AgentExtensionHandle",
    ),
    "Artifact": ("msgflux.nn.extensions.artifact", "Artifact"),
    "ArtifactExtension": ("msgflux.nn.extensions.artifact", "ArtifactExtension"),
    "ArtifactReferenceRenderer": (
        "msgflux.nn.extensions.artifact",
        "ArtifactReferenceRenderer",
    ),
    "ArtifactRegistry": ("msgflux.nn.extensions.artifact", "ArtifactRegistry"),
    "CONTEXT_COMPACTION_CAPABILITY": (
        "msgflux.nn.extensions.compaction",
        "CONTEXT_COMPACTION_CAPABILITY",
    ),
    "CompactionExtension": (
        "msgflux.nn.extensions.compaction",
        "CompactionExtension",
    ),
    "CompactionPolicy": (
        "msgflux.nn.extensions.compaction",
        "CompactionPolicy",
    ),
    "DefaultToolFeedbackExtension": (
        "msgflux.nn.extensions.feedback",
        "DefaultToolFeedbackExtension",
    ),
    "CurrentDateExtension": (
        "msgflux.nn.extensions.prompt",
        "CurrentDateExtension",
    ),
    "FewShotExamplesExtension": (
        "msgflux.nn.extensions.prompt",
        "FewShotExamplesExtension",
    ),
    "SkillsExtension": ("msgflux.nn.extensions.skills", "SkillsExtension"),
    "ToolUsageGuidanceExtension": (
        "msgflux.nn.extensions.prompt",
        "ToolUsageGuidanceExtension",
    ),
    "BackgroundTasksExtension": (
        "msgflux.nn.extensions.tool_library",
        "BackgroundTasksExtension",
    ),
    "MCPServersExtension": (
        "msgflux.nn.extensions.tool_library",
        "MCPServersExtension",
    ),
    "ToolLibraryExtension": (
        "msgflux.nn.extensions.tool_library",
        "ToolLibraryExtension",
    ),
    "ToolLibraryExtensionHandle": (
        "msgflux.nn.extensions.tool_library",
        "ToolLibraryExtensionHandle",
    ),
    "ToolSearchExtension": (
        "msgflux.nn.extensions.tool_library",
        "ToolSearchExtension",
    ),
    "ToolDispatch": ("msgflux.nn.modules.tool_runtime", "ToolDispatch"),
    "ToolContextProvider": (
        "msgflux.nn.modules.tool_runtime",
        "ToolContextProvider",
    ),
    "ToolPolicy": ("msgflux.nn.modules.tool_runtime", "ToolPolicy"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
