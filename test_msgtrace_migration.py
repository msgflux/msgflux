"""
Simple test to verify msgtrace-sdk migration.

This script tests basic telemetry functionality after migrating
from built-in OpenTelemetry to msgtrace-sdk.
"""

import os

# Configure telemetry
os.environ["MSGFLUX_TELEMETRY_ENABLED"] = "true"
os.environ["MSGFLUX_TELEMETRY_EXPORTER"] = "console"

from msgflux.telemetry import Spans
from msgtrace.sdk import MsgTraceAttributes


def test_basic_span():
    """Test basic span creation."""
    print("\n=== Test 1: Basic Span ===")
    with Spans.span_context("test_operation"):
        MsgTraceAttributes.set_operation_name("chat")
        MsgTraceAttributes.set_model("gpt-4")
        MsgTraceAttributes.set_usage(input_tokens=100, output_tokens=50)
        print("✓ Basic span created successfully")


def test_flow_and_module():
    """Test flow and module spans."""
    print("\n=== Test 2: Flow and Module ===")
    with Spans.init_flow("user_query_flow"):
        MsgTraceAttributes.set_workflow_name("test_workflow")

        with Spans.init_module("llm_call"):
            MsgTraceAttributes.set_module_name("openai_chat")
            MsgTraceAttributes.set_module_type("LLM")
            MsgTraceAttributes.set_model("gpt-4")
            print("✓ Flow and module spans created successfully")


@Spans.instrument()
def test_decorator():
    """Test decorator instrumentation."""
    print("\n=== Test 3: Decorator ===")
    MsgTraceAttributes.set_operation_name("tool")
    MsgTraceAttributes.set_tool_name("test_tool")
    print("✓ Decorator instrumentation works")


async def test_async_decorator():
    """Test async decorator."""
    print("\n=== Test 4: Async Decorator ===")

    @Spans.ainstrument()
    async def async_function():
        MsgTraceAttributes.set_operation_name("chat")
        return "async result"

    result = await async_function()
    print(f"✓ Async decorator works: {result}")


def test_tool_attributes():
    """Test msgflux-specific tool attributes."""
    print("\n=== Test 5: Tool Attributes ===")
    with Spans.span_context("tool_execution"):
        MsgTraceAttributes.set_tool_name("search_db")
        MsgTraceAttributes.set_tool_execution_type("local")
        MsgTraceAttributes.set_tool_protocol("mcp")
        MsgTraceAttributes.set_tool_call_arguments({"query": "test", "limit": 10})
        MsgTraceAttributes.set_tool_response({"results": ["a", "b", "c"]})
        print("✓ Tool attributes set successfully")


def test_agent_attributes():
    """Test msgflux-specific agent attributes."""
    print("\n=== Test 6: Agent Attributes ===")
    with Spans.span_context("agent_execution"):
        MsgTraceAttributes.set_agent_name("research_agent")
        MsgTraceAttributes.set_agent_id("agent_001")
        MsgTraceAttributes.set_agent_type("autonomous")
        MsgTraceAttributes.set_agent_response({"status": "completed", "result": "data"})
        print("✓ Agent attributes set successfully")


def test_module_attributes():
    """Test module attributes (new in msgtrace-sdk)."""
    print("\n=== Test 7: Module Attributes ===")
    with Spans.init_module("vector_search"):
        MsgTraceAttributes.set_module_name("faiss_retriever")
        MsgTraceAttributes.set_module_type("Retriever")
        print("✓ Module attributes set successfully")


if __name__ == "__main__":
    import asyncio

    print("\n" + "=" * 60)
    print("  msgFlux → msgtrace-sdk Migration Test")
    print("=" * 60)

    # Run sync tests
    test_basic_span()
    test_flow_and_module()
    test_decorator()
    test_tool_attributes()
    test_agent_attributes()
    test_module_attributes()

    # Run async test
    asyncio.run(test_async_decorator())

    print("\n" + "=" * 60)
    print("  ✅ All tests passed!")
    print("=" * 60 + "\n")
