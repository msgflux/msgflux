"""Compatibility facade for the fragmented ToolLibrary runtime."""

from msgflux.nn.modules.tool.definitions import ToolDefinition, ToolDefinitionCompiler
from msgflux.nn.modules.tool.execution import (
    AfterToolPolicy,
    BeforeDispatchPolicy,
    BeforeToolPolicy,
    DispatchRequest,
    ToolExecutionPlan,
    ToolRuntimeContext,
)
from msgflux.nn.modules.tool.extensions import (
    BackgroundDispatch,
    ContextRequest,
    DetachedDispatch,
    ForegroundDispatch,
    RuntimeContextProvider,
    ToolContextProvider,
    ToolDispatch,
    ToolExtension,
    ToolExtensionHandle,
    ToolExtensionRegistry,
    ToolPolicy,
)
from msgflux.nn.modules.tool.registry import ToolRegistry
from msgflux.tools.catalog import (
    NativeToolBinding,
    ToolCatalogEntry,
    ToolCatalogView,
    ToolChoice,
    ToolRef,
)
from msgflux.tools.runtime import FeedbackSpec, ToolError, ToolIntent, ToolOutcome
from msgflux.tools.specs import ContextBinding, ContextSpec, DispatchSpec, LoadingSpec

__all__ = [
    "AfterToolPolicy",
    "BackgroundDispatch",
    "BeforeDispatchPolicy",
    "BeforeToolPolicy",
    "ContextBinding",
    "ContextRequest",
    "ContextSpec",
    "DetachedDispatch",
    "DispatchRequest",
    "DispatchSpec",
    "FeedbackSpec",
    "ForegroundDispatch",
    "LoadingSpec",
    "NativeToolBinding",
    "RuntimeContextProvider",
    "ToolCatalogEntry",
    "ToolCatalogView",
    "ToolChoice",
    "ToolContextProvider",
    "ToolDefinition",
    "ToolDefinitionCompiler",
    "ToolDispatch",
    "ToolError",
    "ToolExecutionPlan",
    "ToolExtension",
    "ToolExtensionHandle",
    "ToolExtensionRegistry",
    "ToolIntent",
    "ToolOutcome",
    "ToolPolicy",
    "ToolRef",
    "ToolRegistry",
    "ToolRuntimeContext",
]
