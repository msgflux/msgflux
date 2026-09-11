"""
Test script to generate traces with local tools for visualization testing
"""

import sys

sys.path.insert(0, "./src")

from typing import Any, Dict, List

from msgflux.nn.modules import ToolLibrary


# Local Tools (simple functions with docstrings)
def calculate_sum(numbers: List[float]) -> float:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)


def get_user_info(user_id: str) -> Dict[str, str]:
    """Get user information by ID."""
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "role": "developer",
    }


def search_database(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search database with query and return results."""
    return [
        {"id": "1", "title": "Result 1", "content": f"Content about {query}"},
        {"id": "2", "title": "Result 2", "content": f"More about {query}"},
        {"id": "3", "title": "Result 3", "content": f"Details on {query}"},
    ][:limit]


def format_response(data: Dict[str, Any], format_type: str = "json") -> str:
    """Format data in specified format."""
    if format_type == "json":
        import json

        return json.dumps(data, indent=2)
    elif format_type == "markdown":
        lines = ["# Data Report\n"]
        for key, value in data.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)
    return str(data)


def validate_input(text: str, min_length: int = 5) -> bool:
    """Validate input text meets requirements."""
    return len(text.strip()) >= min_length


# Test function
def test_tool_library():
    """Test ToolLibrary to generate traces."""

    print("🧪 Testing ToolLibrary...")

    # Create tool library
    tool_library = ToolLibrary(
        name="test",
        tools=[
            calculate_sum,
            get_user_info,
            search_database,
            format_response,
            validate_input,
        ],
    )

    print(f"✓ ToolLibrary created with {len(tool_library.get_tool_names())} tools")
    print(f"  Tools: {', '.join(tool_library.get_tool_names())}")

    # Test tool calls through the library
    print("\n📞 Executing tool calls...")

    # Call 1: Calculate sum
    tool_callings_1 = [
        ("call_1", "calculate_sum", {"numbers": [10.5, 20.3, 30.7, 15.2]})
    ]
    result1 = tool_library(tool_callings_1)
    print(f"✓ calculate_sum: {result1.tool_calls[0].result}")

    # Call 2: Get user info
    tool_callings_2 = [("call_2", "get_user_info", {"user_id": "user_123"})]
    result2 = tool_library(tool_callings_2)
    print(f"✓ get_user_info: {result2.tool_calls[0].result}")

    # Call 3: Search database
    tool_callings_3 = [
        ("call_3", "search_database", {"query": "artificial intelligence", "limit": 3})
    ]
    result3 = tool_library(tool_callings_3)
    print(f"✓ search_database: {len(result3.tool_calls[0].result)} results")

    # Call 4: Format response
    user_data = result2.tool_calls[0].result
    tool_callings_4 = [
        ("call_4", "format_response", {"data": user_data, "format_type": "markdown"})
    ]
    result4 = tool_library(tool_callings_4)
    print("✓ format_response: Markdown formatted")

    # Call 5: Validate input
    tool_callings_5 = [
        ("call_5", "validate_input", {"text": "Hello World!", "min_length": 5})
    ]
    result5 = tool_library(tool_callings_5)
    print(f"✓ validate_input: {result5.tool_calls[0].result}")

    # Call 6: Multiple tools at once
    print("\n📞 Executing multiple tools in parallel...")
    tool_callings_multi = [
        ("call_6a", "calculate_sum", {"numbers": [1, 2, 3, 4, 5]}),
        ("call_6b", "validate_input", {"text": "Test", "min_length": 3}),
        ("call_6c", "get_user_info", {"user_id": "user_456"}),
    ]
    result_multi = tool_library(tool_callings_multi)
    print(f"✓ Executed {len(result_multi.tool_calls)} tools in parallel")

    print("\n✅ All tools tested successfully!")

    return tool_library


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MsgTrace Tools Visualization Test")
    print("=" * 60)
    print()

    # Test tool library
    tool_library = test_tool_library()

    print("\n" + "=" * 60)
    print("✨ Test completed!")
    print("=" * 60)
    print()
    print("📊 View traces at: http://localhost:8000")
    print("   - Click on a trace to see details")
    print("   - Switch to 'Tools' tab to see the new visualization")
    print()
