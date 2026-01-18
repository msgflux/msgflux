"""msgflux.telemetry - msgtrace-sdk integration.

Provides telemetry capabilities through msgtrace-sdk with msgflux-specific
decorators for detailed tool and agent instrumentation.
"""

from msgtrace.sdk import Spans

from msgflux.telemetry.span import (
    aset_agent_attributes,
    aset_tool_attributes,
    set_agent_attributes,
    set_tool_attributes,
)

__all__ = [
    "Spans",
    "aset_agent_attributes",
    "aset_tool_attributes",
    "set_agent_attributes",
    "set_tool_attributes",
]
