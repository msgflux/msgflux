"""Tools module for msgflux.

This module provides tool-related functionality including
ToolFlowControl for managing tool execution flow.
"""

from msgflux.generation.control_flow import ToolFlowControl
from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.definitions import ToolDefinitions
from msgflux.tools.guidance import BUILTIN_TOOL_USAGE_GUIDANCE, apply_tool_guidance
from msgflux.tools.handles import ToolHandle
from msgflux.tools.types import (
    Hidden,
    ToolBackground,
    ToolBucket,
    ToolLibraryOperator,
)

__all__ = [
    "BUILTIN_TOOL_USAGE_GUIDANCE",
    "Hidden",
    "ToolBackground",
    "ToolDefinitions",
    "ToolBucket",
    "ToolFlowControl",
    "ToolHandle",
    "ToolLibraryOperator",
    "ToolMetadata",
    "apply_tool_guidance",
]
