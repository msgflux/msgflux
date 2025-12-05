"""
msgflux telemetry span - msgtrace-sdk integration.

Provides Spans from msgtrace-sdk with msgflux-specific
tool and agent decorators for detailed telemetry capture.
"""

import hashlib
from functools import wraps
from typing import Callable, Optional

import msgspec
from msgtrace.sdk import MsgTraceAttributes
from opentelemetry import trace

from msgflux.envs import envs
from msgflux.models.response import ModelStreamResponse


def set_tool_attributes(execution_type: str, protocol: Optional[str] = None) -> Callable:
    """
    Decorator to set detailed tool telemetry attributes for sync methods.

    This msgflux-specific decorator captures:
    - Tool name, description, type
    - Tool call ID (if provided)
    - Tool call arguments (JSON encoded)
    - Tool execution type (local/remote)
    - Tool protocol (mcp/a2a/etc)
    - Tool response (JSON encoded)

    Args:
        execution_type: Execution type ('local' or 'remote')
        protocol: Protocol ('mcp', 'a2a', etc)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, **kwargs):
            # Extract tool_call_id from kwargs (passed by ToolLibrary)
            tool_call_id = kwargs.pop("tool_call_id", None)

            span = trace.get_current_span()
            if span.is_recording():
                # Set operation and tool metadata
                MsgTraceAttributes.set_operation_name("tool")
                MsgTraceAttributes.set_tool_name(self.name)

                if hasattr(self, "description") and self.description:
                    MsgTraceAttributes.set_tool_description(self.description)

                # Set tool call ID if available
                if tool_call_id:
                    MsgTraceAttributes.set_tool_call_id(tool_call_id)

                # Set tool type
                span.set_attribute("gen_ai.tool.type", "function")

                # Set msgflux-specific attributes
                MsgTraceAttributes.set_tool_execution_type(execution_type)
                if protocol:
                    MsgTraceAttributes.set_tool_protocol(protocol)

                # Capture arguments (without tool_call_id)
                try:
                    encoded_args = msgspec.json.encode(kwargs)
                    MsgTraceAttributes.set_tool_call_arguments(encoded_args.decode())
                except (TypeError, ValueError):
                    pass

            # Execute the actual function
            response = func(self, **kwargs)

            # Capture result
            span = trace.get_current_span()
            if span.is_recording() and envs.telemetry_capture_tool_call_responses:
                try:
                    if isinstance(response, (str, int, float, bool)):
                        MsgTraceAttributes.set_tool_response(str(response))
                    else:
                        encoded_response = msgspec.json.encode(response)
                        MsgTraceAttributes.set_tool_response(encoded_response.decode())
                except (TypeError, ValueError):
                    pass

            return response

        return wrapper

    return decorator


def aset_tool_attributes(execution_type: str, protocol: Optional[str] = None) -> Callable:
    """
    Decorator to set detailed tool telemetry attributes for async methods.

    Args:
        execution_type: Execution type ('local' or 'remote')
        protocol: Protocol ('mcp', 'a2a', etc)

    Returns:
        Decorated async function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Extract tool_call_id from kwargs
            tool_call_id = kwargs.pop("tool_call_id", None)

            span = trace.get_current_span()
            if span.is_recording():
                # Set operation and tool metadata
                MsgTraceAttributes.set_operation_name("tool")
                MsgTraceAttributes.set_tool_name(self.name)

                if hasattr(self, "description") and self.description:
                    MsgTraceAttributes.set_tool_description(self.description)

                # Set tool call ID if available
                if tool_call_id:
                    MsgTraceAttributes.set_tool_call_id(tool_call_id)

                # Set tool type
                span.set_attribute("gen_ai.tool.type", "function")

                # Set msgflux-specific attributes
                MsgTraceAttributes.set_tool_execution_type(execution_type)
                if protocol:
                    MsgTraceAttributes.set_tool_protocol(protocol)

                # Capture arguments
                try:
                    encoded_args = msgspec.json.encode(kwargs)
                    MsgTraceAttributes.set_tool_call_arguments(encoded_args.decode())
                except (TypeError, ValueError):
                    pass

            # Execute the actual function
            response = await func(self, *args, **kwargs)

            # Capture result
            span = trace.get_current_span()
            if span.is_recording() and envs.telemetry_capture_tool_call_responses:
                try:
                    if isinstance(response, (str, int, float, bool)):
                        MsgTraceAttributes.set_tool_response(str(response))
                    else:
                        encoded_response = msgspec.json.encode(response)
                        MsgTraceAttributes.set_tool_response(encoded_response.decode())
                except (TypeError, ValueError):
                    pass

            return response

        return wrapper

    return decorator


def set_agent_attributes(func: Callable) -> Callable:
    """
    Decorator to set detailed agent telemetry attributes for sync methods.

    Captures:
    - Agent name and generated ID (hash of name)
    - Agent description
    - Agent response (if not streaming)

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        span = trace.get_current_span()
        if span.is_recording():
            # Set agent metadata
            if hasattr(self, "name") and self.name:
                MsgTraceAttributes.set_agent_name(self.name)

                # Generate agent ID from name hash
                agent_id = f"asst_{hashlib.sha256(self.name.encode()).hexdigest()[:24]}"
                MsgTraceAttributes.set_agent_id(agent_id)

            # Set agent description if available
            if hasattr(self, "description") and self.description:
                span.set_attribute("gen_ai.agent.description", self.description)

        # Execute the actual function
        response = func(self, *args, **kwargs)

        # Capture result (avoid streaming responses)
        span = trace.get_current_span()
        if span.is_recording() and not isinstance(response, ModelStreamResponse):
            MsgTraceAttributes.set_agent_response(response)

        return response

    return wrapper


def aset_agent_attributes(func: Callable) -> Callable:
    """
    Decorator to set detailed agent telemetry attributes for async methods.

    Args:
        func: Async function to decorate

    Returns:
        Decorated async function
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        span = trace.get_current_span()
        if span.is_recording():
            # Set agent metadata
            if hasattr(self, "name") and self.name:
                MsgTraceAttributes.set_agent_name(self.name)

                # Generate agent ID from name hash
                agent_id = f"asst_{hashlib.sha256(self.name.encode()).hexdigest()[:24]}"
                MsgTraceAttributes.set_agent_id(agent_id)

            # Set agent description if available
            if hasattr(self, "description") and self.description:
                span.set_attribute("gen_ai.agent.description", self.description)

        # Execute the actual function
        response = await func(self, *args, **kwargs)

        # Capture result (avoid streaming responses)
        span = trace.get_current_span()
        if span.is_recording() and not isinstance(response, ModelStreamResponse):
            MsgTraceAttributes.set_agent_response(response)

        return response

    return wrapper
