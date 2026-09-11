from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Union

from msgflux.models.tool_transport import render_native_output, transport_adapter
from msgflux.tools.runtime import ToolIntent, ToolOutcome
from msgflux.utils.chat import ChatBlock
from msgflux.utils.msgspec import msgspec_dumps, msgspec_loads


class ToolCallAggregator:
    """Accumulate provider tool-call deltas and encode their continuation.

    The accumulator is created by the Model, so it owns the API-specific
    continuation format. Runtime consumers deal only with ToolIntent and
    ToolOutcome.
    """

    def __init__(
        self,
        reasoning: Optional[str] = None,
        *,
        api_mode: Literal["chat_completions", "responses"] = "chat_completions",
    ):
        if api_mode not in {"chat_completions", "responses"}:
            raise ValueError(f"Unsupported tool-call API mode: `{api_mode}`")
        self.reasoning = reasoning
        self.api_mode = api_mode
        self.tool_calls = OrderedDict()
        self.native_calls = {}

    def process_native(self, index: int, item: dict, adapter, name: str):
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Native tool call requires a call_id")
        if call_id in self.native_calls:
            return
        arguments, metadata = adapter.decode(item, name)
        transport_adapter(metadata)
        self.process(index, call_id, name, msgspec_dumps(arguments))
        self.native_calls[call_id] = metadata
        item.setdefault("metadata", {})["tool_transport"] = metadata.copy()

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

    def get_calls(self) -> List[tuple[str, str, Any]]:
        """Returns the function name and arguments in a dict format."""
        tool_callings = []
        for call in self.tool_calls.values():
            arguments = call["arguments"].strip()
            if arguments:
                arguments = msgspec_loads(arguments)
            tool_callings.append((call["id"], call["name"], arguments))
        return tool_callings

    def get_intents(self) -> tuple[ToolIntent, ...]:
        """Decode accumulated provider calls into canonical runtime intents."""
        return tuple(
            ToolIntent(id=call_id, name=name, arguments=arguments or {})
            for call_id, name, arguments in self.get_calls()
        )

    @staticmethod
    def _outcome_output(outcome: ToolOutcome) -> Any:
        if outcome.error is not None:
            if outcome.status == "interrupted":
                return {
                    "status": "interrupted",
                    "reason": "user_requested_stop",
                    "message": outcome.error.message,
                }
            return outcome.error.message
        return outcome.result

    def render_outcomes(
        self,
        outcomes: List[ToolOutcome] | tuple[ToolOutcome, ...],
    ) -> List[Dict[str, Any]]:
        """Encode canonical outcomes for the API that produced these calls."""
        by_id = {outcome.intent_id: outcome for outcome in outcomes}
        missing = [
            call["id"] for call in self.tool_calls.values() if call["id"] not in by_id
        ]
        if missing:
            formatted = ", ".join(f"`{call_id}`" for call_id in missing)
            raise ValueError(f"Missing tool outcomes for call ids: {formatted}")

        if self.api_mode == "responses":
            rendered = []
            for call in self.tool_calls.values():
                outcome = by_id[call["id"]]
                native = self.native_calls.get(call["id"])
                if native is not None:
                    rendered.append(
                        render_native_output(
                            call["id"],
                            outcome.result,
                            native,
                            error=outcome.error.message if outcome.error else None,
                        )
                    )
                    continue
                output = self._outcome_output(outcome)
                if not isinstance(output, str):
                    output = msgspec_dumps(output)
                item = {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": output,
                }
                rendered.append(item)
            return rendered

        self.insert_results(
            {outcome.intent_id: self._outcome_output(outcome) for outcome in outcomes}
        )
        return self.get_messages()

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
        1. The first message contains all the function call requests
           (with reasoning in <think> tags if present).
        2. Subsequent messages insert the results of the functions, one at a time.
        """
        # First message: function calls (reasoning embedded in content with <think>)
        tool_calls = [
            ChatBlock.tool_call(call["id"], call["name"], call["arguments"])
            for call in self.tool_calls.values()
        ]
        messages = [ChatBlock.assist_tool_calls(tool_calls, reasoning=self.reasoning)]

        # Adding the results of function calls as separate messages
        for call in self.tool_calls.values():
            if call["result"] is not None:
                if not isinstance(call["result"], str):  # convert to str
                    call["result"] = msgspec_dumps(call["result"])
                messages.append(ChatBlock.tool(call["id"], call["result"]))

        return messages
