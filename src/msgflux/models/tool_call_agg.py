from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import msgspec


class ToolCallAggregator:
    def __init__(self, reasoning: Optional[str] = None):
        self.reasoning = reasoning
        self.tool_calls = OrderedDict()

    def process(self, call_index: int, tool_id: str, name: str, arguments: str):
        """Add tool call.

        In stream mode, the tool call arguments are generated each new chunk.

        Args:
            call_index: Call index in tool call sequence
            arguments: Tool call argument
            tool_id: Call id
            name: Tool name
        """
        # If call_index exists, add arguments
        if call_index in self.tool_calls:
            current_call = self.tool_calls[call_index]
            current_call["arguments"] += arguments
            # If name and id are not filled in, update
            if not current_call["id"] and tool_id:
                current_call["id"] = tool_id
            if not current_call["name"] and name:
                current_call["name"] = name
        else:
            # Init a new function call
            self.tool_calls[call_index] = {
                "id": tool_id or None,  # Can be filled in later
                "name": name,
                "arguments": arguments,
            }

    def get_calls(self) -> Dict[str, str]:
        """Returns the function name and arguments in a dict format."""
        tool_callings = []
        for call in self.tool_calls.values():
            arguments = call["arguments"].strip()
            if arguments:
                arguments = msgspec.json.decode(arguments.encode())
            tool_callings.append((call["id"], call["name"], arguments))
        return tool_callings

    def insert_results(self, tool_results: Dict[str, Union[str, None]]):
        """Inserts the results of the called functions into the tool_calls dict.

        Args:
            tool_results:
                Dictionary where the key is the tool id and the value is the result.
        """
        for tool_id, result in tool_results.items():
            for call in self.tool_calls.values():
                if call["id"] == tool_id:
                    call["result"] = result

    def get_messages(self) -> List[Dict[str, Any]]:
        """Generates a list of messages to send to the model:
        1. The first message contains all the function call requests.
        2. Subsequent messages insert the results of the functions, one at a time.
        """
        # First message: function calls
        tool_calls = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"arguments": call["arguments"], "name": call["name"]},
            }
            for call in self.tool_calls.values()
        ]
        messages = [{"role": "assistant", "tool_calls": tool_calls}]

        # Adding the results of function calls as separate messages
        for call in self.tool_calls.values():
            if call["result"] is not None:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": call["result"],
                    }
                )

        if self.reasoning is not None:
            messages.insert(0, {"role": "assistant", "content": self.reasoning})

        return messages
