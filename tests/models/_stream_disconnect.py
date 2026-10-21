"""Synthetic SSE frames shared by stream-drop and service fault tests."""

import msgspec


def _sse(*payloads):
    return b"".join(b"data: " + msgspec.json.encode(p) + b"\n\n" for p in payloads)


def _chat_chunk(delta, finish_reason=None):
    return {
        "id": "synthetic-completion",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def partial_text(api_mode):
    if api_mode == "responses":
        return _sse(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": "partial-before-drop",
            }
        )
    return _sse(_chat_chunk({"role": "assistant", "content": "partial-before-drop"}))


def tool_call(api_mode, *, complete):
    call_id = "effect-call" if complete else "incomplete-effect-call"
    arguments = "{}" if complete else "{"
    if api_mode == "responses":
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": "record_effect",
            "arguments": "",
        }
        payloads = [
            {"type": "response.output_item.added", "output_index": 1, "item": item},
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "delta": arguments,
            },
        ]
        if complete:
            payloads.append(
                {
                    "type": "response.completed",
                    "response": {"id": "synthetic-response", "status": "completed"},
                }
            )
        return _sse(*payloads)
    frame = _chat_chunk(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "record_effect", "arguments": arguments},
                }
            ],
        },
        "tool_calls" if complete else None,
    )
    return _sse(frame) + (b"data: [DONE]\n\n" if complete else b"")
