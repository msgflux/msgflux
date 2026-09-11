from importlib import import_module

_MODULE_EXPORTS = [
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
    "ToolExecutionPlan",
    "ToolLibrary",
    "Transcriber",
]

__all__ = [
    "AgentExtension",
    "AgentExtensionHandle",
    "Artifact",
    "ArtifactExtension",
    "ArtifactReferenceRenderer",
    "ArtifactRegistry",
    "BackgroundTasksExtension",
    "CompactionExtension",
    "CompactionPolicy",
    "CONTEXT_COMPACTION_CAPABILITY",
    "CurrentDateExtension",
    "FewShotExamplesExtension",
    "ContextBinding",
    "Parameter",
    "functional",
    "modules",
    "parameter",
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
    "SkillsExtension",
    "MCPServersExtension",
    "ToolLibraryExtension",
    "ToolLibraryExtensionHandle",
    "ToolDispatch",
    "ToolContextProvider",
    "ToolPolicy",
    "ToolSearchExtension",
    "DefaultToolFeedbackExtension",
    "ToolTurnLimitExtension",
    "ToolUsageGuidanceExtension",
    "Tool",
    "ToolExecutionPlan",
    "ToolLibrary",
    "Transcriber",
]


def __getattr__(name: str):
    if name in {"functional", "modules", "parameter"}:
        value = import_module(f"msgflux.nn.{name}")
    elif name in {
        "AgentExtension",
        "AgentExtensionHandle",
        "Artifact",
        "ArtifactExtension",
        "ArtifactReferenceRenderer",
        "ArtifactRegistry",
        "BackgroundTasksExtension",
        "CompactionExtension",
        "CompactionPolicy",
        "CONTEXT_COMPACTION_CAPABILITY",
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
        "DefaultToolFeedbackExtension",
        "ToolTurnLimitExtension",
        "ToolUsageGuidanceExtension",
    }:
        value = getattr(import_module("msgflux.nn.extensions"), name)
    elif name == "Parameter":
        value = getattr(import_module("msgflux.nn.parameter"), name)
    elif name == "ContextBinding":
        value = getattr(import_module("msgflux.tools.specs"), name)
    elif name in _MODULE_EXPORTS:
        value = getattr(import_module("msgflux.nn.modules"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
