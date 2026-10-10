"""Tools module for msgflux.

This module provides tool-related functionality including
ToolFlowControl for managing tool execution flow.
"""

from msgflux.generation.control_flow import ToolFlowControl
from msgflux.tools.catalog import (
    ToolCatalogEntry,
    ToolCatalogView,
    ToolChoice,
    ToolRef,
)
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.handles import ToolBucketHandle, ToolLibraryHandle
from msgflux.tools.runtime import FeedbackSpec, ToolError, ToolIntent, ToolOutcome
from msgflux.tools.types import (
    Hidden,
    ToolBackground,
    ToolBucket,
    ToolBucketEntry,
    ToolLibraryOperator,
)

__all__ = [
    "FeedbackSpec",
    "Hidden",
    "ToolBackground",
    "ToolCatalog",
    "ToolCatalogEntry",
    "ToolCatalogView",
    "ToolChoice",
    "ToolError",
    "ToolIntent",
    "ToolOutcome",
    "ToolRef",
    "ToolSpec",
    "ToolBucket",
    "ToolBucketEntry",
    "ToolBucketHandle",
    "ToolFlowControl",
    "ToolLibraryHandle",
    "ToolLibraryOperator",
]
