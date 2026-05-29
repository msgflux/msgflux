"""ChatMessages — container for chat history with serialization support."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, List, Mapping

from msgflux.context import (
    _CURRENT_NAMESPACE,
    _CURRENT_SESSION_ID,
    get_session_context,
    session_context,
)
from msgflux.core.examples import Example
from msgflux.data.types import Audio, File, Image, MediaType, Video
from msgflux.utils.msgspec import msgspec_dumps


class ChatMessages:
    """Container for chat history with optional turn tracking.

    The class behaves like a mutable list of normalized chat items while also
    carrying `session_id`, turn metadata, and state serialization helpers used
    by durable agents.
    """

    def __init__(
        self,
        items: Iterable[Mapping[str, Any]] | None = None,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ):
        self._items: List[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}
        self.reasoning_content: str | None = None
        self.reasoning_text: str | None = None
        self.response_id: str | None = None
        self.session_id: str | None = (
            session_id if session_id is not None else _CURRENT_SESSION_ID.get()
        )
        self.namespace: str | None = (
            namespace if namespace is not None else _CURRENT_NAMESPACE.get()
        )
        self._turns: List[dict[str, Any]] = []
        self._active_turn_index: int | None = None

        if items is not None:
            self.extend(items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        normalized_items = self._normalize_item(item)
        if not normalized_items:
            raise ValueError("`item` generated an empty normalized representation")
        if len(normalized_items) == 1:
            self._items[index] = normalized_items[0]
        else:
            self._items[index : index + 1] = normalized_items

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        preview = self._items[-3:]
        return (
            "ChatMessages("
            f"size={len(self._items)}, "
            f"turns={len(self._turns)}, "
            f"session_id={self.session_id!r}, "
            f"namespace={self.namespace!r}, "
            f"preview={preview})"
        )

    def append(self, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        normalized_items = self._normalize_item(item)
        self._items.extend(normalized_items)

    def insert(self, index: int, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        normalized_items = self._normalize_item(item)
        self._items[index:index] = normalized_items

    def insert_before_active_turn(self, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        if self._active_turn_index is None:
            self.append(item)
            return

        normalized_items = self._normalize_item(item)
        start_item_index = self._turns[self._active_turn_index].get("start_item_index")
        if not isinstance(start_item_index, int):
            self.append(item)
            return

        self._items[start_item_index:start_item_index] = normalized_items
        self._turns[self._active_turn_index]["start_item_index"] = (
            start_item_index + len(normalized_items)
        )

    def extend(self, items: Iterable[Mapping[str, Any]]) -> None:
        if isinstance(items, ChatMessages):
            items = items._items
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
            normalized_items = self._normalize_item(item)
            self._items.extend(normalized_items)

    def copy(self) -> ChatMessages:
        copied = ChatMessages(
            self._items,
            session_id=self.session_id,
            namespace=self.namespace,
        )
        copied.metadata = deepcopy(self.metadata)
        copied.reasoning_content = self.reasoning_content
        copied.reasoning_text = self.reasoning_text
        copied.response_id = self.response_id
        copied._turns = deepcopy(self._turns)
        copied._active_turn_index = self._active_turn_index
        return copied

    session_context = staticmethod(session_context)
    get_session_context = staticmethod(get_session_context)

    def configure_session(
        self,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> None:
        resolved_session_id = (
            session_id
            if session_id is not None
            else (
                self.session_id
                if self.session_id is not None
                else _CURRENT_SESSION_ID.get()
            )
        )
        self.session_id = resolved_session_id
        if namespace is not None:
            self.namespace = namespace
        elif self.namespace is None:
            self.namespace = _CURRENT_NAMESPACE.get()

    def begin_turn(
        self,
        *,
        inputs: Any = None,
        context_inputs: Any = None,
        vars: Mapping[str, Any] | None = None,  # noqa: A002
        session_id: str | None = None,
        namespace: str | None = None,
        turn_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if self._active_turn_index is not None:
            self.end_turn(status="interrupted")

        self.configure_session(session_id=session_id, namespace=namespace)

        turn_index = len(self._turns)
        turn_identifier = (
            turn_id
            if isinstance(turn_id, str) and turn_id
            else f"turn_{turn_index + 1}"
        )
        turn_record = {
            "turn_id": turn_identifier,
            "index": turn_index,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "started_at": self._utcnow_iso(),
            "ended_at": None,
            "status": "in_progress",
            "start_item_index": len(self._items),
            "end_item_index": None,
            "inputs": self._safe_copy(inputs),
            "context_inputs": self._safe_copy(context_inputs),
            "vars": self._safe_copy(dict(vars or {})),
            "assistant_output": None,
            "response_type": None,
            "response_metadata": {},
            "metadata": self._safe_copy(dict(metadata or {})),
        }
        self._turns.append(turn_record)
        self._active_turn_index = turn_index

        self.append(
            {
                "type": "turn_marker",
                "event": "turn_start",
                "turn_id": turn_identifier,
                "turn_index": turn_index,
                "session_id": self.session_id,
                "namespace": self.namespace,
                "timestamp": turn_record["started_at"],
            }
        )

        return turn_identifier

    def end_turn(
        self,
        *,
        assistant_output: Any = None,
        response_type: str | None = None,
        response_metadata: Mapping[str, Any] | None = None,
        status: str = "completed",
    ) -> Mapping[str, Any] | None:
        if self._active_turn_index is None:
            return None

        turn_record = self._turns[self._active_turn_index]
        turn_record["assistant_output"] = self._safe_copy(assistant_output)
        turn_record["response_type"] = response_type
        if response_metadata is not None:
            turn_record["response_metadata"] = self._safe_copy(dict(response_metadata))
        turn_record["status"] = status
        turn_record["ended_at"] = self._utcnow_iso()
        self._active_turn_index = None

        self.append(
            {
                "type": "turn_marker",
                "event": "turn_end",
                "turn_id": turn_record["turn_id"],
                "turn_index": turn_record["index"],
                "session_id": self.session_id,
                "namespace": turn_record["namespace"],
                "timestamp": turn_record["ended_at"],
                "status": status,
            }
        )
        turn_record["end_item_index"] = len(self._items) - 1

        return deepcopy(turn_record)

    @property
    def turns(self) -> List[dict[str, Any]]:
        return deepcopy(self._turns)

    def get_active_turn(self) -> Mapping[str, Any] | None:
        if self._active_turn_index is None:
            return None
        return deepcopy(self._turns[self._active_turn_index])

    def get_active_turn_size(self) -> int:
        if self._active_turn_index is None:
            return 0
        start_item_index = self._turns[self._active_turn_index].get("start_item_index")
        if not isinstance(start_item_index, int):
            return 0
        return len(self._items) - start_item_index

    def fork(self, *, upto_turn: int | None = None) -> ChatMessages:
        if upto_turn is None:
            return self.copy()

        if not isinstance(upto_turn, int):
            raise TypeError(
                f"`upto_turn` must be int or None, given `{type(upto_turn)}`"
            )

        if upto_turn < 0:
            forked = ChatMessages(
                session_id=self.session_id,
                namespace=self.namespace,
            )
            forked.metadata = deepcopy(self.metadata)
            forked.reasoning_content = self.reasoning_content
            forked.reasoning_text = self.reasoning_text
            forked.response_id = self.response_id
            return forked

        if upto_turn >= len(self._turns):
            return self.copy()

        selected_turn = self._turns[upto_turn]
        end_item_index = selected_turn.get("end_item_index")
        if not isinstance(end_item_index, int):
            end_item_index = len(self._items) - 1

        forked = ChatMessages(
            self._items[: end_item_index + 1],
            session_id=self.session_id,
            namespace=self.namespace,
        )
        forked.metadata = deepcopy(self.metadata)
        forked.reasoning_content = self.reasoning_content
        forked.reasoning_text = self.reasoning_text
        forked.response_id = self.response_id
        forked._turns = deepcopy(self._turns[: upto_turn + 1])
        forked._active_turn_index = None
        return forked

    def to_examples(
        self,
        *,
        include_history: bool = True,
        history_key: str = "history",
        output_key: str = "response",
    ) -> List[Example]:
        examples: List[Example] = []

        for turn in self._turns:
            if turn.get("assistant_output") is None:
                continue

            turn_inputs = turn.get("inputs")
            if isinstance(turn_inputs, Mapping):
                example_inputs: Mapping[str, Any] = deepcopy(dict(turn_inputs))
            elif turn_inputs is None:
                example_inputs = {}
            else:
                example_inputs = {"input": self._safe_copy(turn_inputs)}
            example_inputs = dict(example_inputs)

            if turn.get("context_inputs") is not None:
                example_inputs["context_inputs"] = self._safe_copy(
                    turn.get("context_inputs")
                )

            vars_payload = turn.get("vars")
            if isinstance(vars_payload, Mapping) and vars_payload:
                example_inputs["vars"] = self._safe_copy(dict(vars_payload))

            if include_history:
                start_item_index = turn.get("start_item_index")
                if not isinstance(start_item_index, int):
                    start_item_index = 0
                history_items = self._items[:start_item_index]
                example_inputs[history_key] = ChatMessages(history_items).to_chatml()

            reasoning = self._extract_turn_reasoning(turn)
            labels = {output_key: self._safe_copy(turn.get("assistant_output"))}
            examples.append(
                Example(
                    inputs=example_inputs,
                    labels=labels,
                    reasoning=reasoning,
                    topic=turn.get("namespace"),
                )
            )

        return examples

    @classmethod
    def from_chatml(cls, messages: Iterable[Mapping[str, Any]]) -> ChatMessages:
        return cls(messages)

    def add_chatml(self, messages: Iterable[Mapping[str, Any]]) -> None:
        self.extend(messages)

    def add_response_items(self, items: Iterable[Mapping[str, Any]]) -> None:
        self.extend(items)

    def add_message(self, role: str, content: Any) -> None:
        if not isinstance(role, str):
            raise TypeError(f"`role` must be str, given `{type(role)}`")
        self.append({"role": role, "content": content})

    def add_user(self, content: Any) -> None:
        self.add_message("user", content)

    def add_user_multimodal(
        self,
        *,
        text: str | None = None,
        media: Mapping[str, Any] | None = None,
        image_block_kwargs: Mapping[str, Any] | None = None,
        video_block_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if media is None:
            self.add_user("" if text is None else text)
            return

        content = self.build_multimodal_content(
            text=text,
            media=media,
            image_block_kwargs=image_block_kwargs,
            video_block_kwargs=video_block_kwargs,
        )
        if content:
            self.add_user(content)

    async def aadd_user_multimodal(
        self,
        *,
        text: str | None = None,
        media: Mapping[str, Any] | None = None,
        image_block_kwargs: Mapping[str, Any] | None = None,
        video_block_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if media is None:
            self.add_user("" if text is None else text)
            return

        content = await self.abuild_multimodal_content(
            text=text,
            media=media,
            image_block_kwargs=image_block_kwargs,
            video_block_kwargs=video_block_kwargs,
        )
        if content:
            self.add_user(content)

    @classmethod
    def build_multimodal_content(
        cls,
        *,
        text: str | None = None,
        media: Mapping[str, Any] | None = None,
        image_block_kwargs: Mapping[str, Any] | None = None,
        video_block_kwargs: Mapping[str, Any] | None = None,
    ) -> List[dict[str, Any]]:
        if media is not None and not isinstance(media, Mapping):
            raise TypeError(f"`media` must be Mapping or None, given `{type(media)}`")

        image_kwargs = dict(image_block_kwargs or {})
        video_kwargs = dict(video_block_kwargs or {})
        content: List[dict[str, Any]] = []

        media_mapping = media or {}
        for media_type in ("image", "audio", "video", "file"):
            media_sources = cls._iter_media_sources(media_mapping.get(media_type))
            for media_source in media_sources:
                formatted_input = cls._build_media_input_sync(
                    media_type,
                    media_source,
                    image_kwargs,
                    video_kwargs,
                )
                if formatted_input:
                    content.append(formatted_input)

        if text not in (None, ""):
            content.append({"type": "text", "text": text})

        return content

    @classmethod
    async def abuild_multimodal_content(
        cls,
        *,
        text: str | None = None,
        media: Mapping[str, Any] | None = None,
        image_block_kwargs: Mapping[str, Any] | None = None,
        video_block_kwargs: Mapping[str, Any] | None = None,
    ) -> List[dict[str, Any]]:
        if media is not None and not isinstance(media, Mapping):
            raise TypeError(f"`media` must be Mapping or None, given `{type(media)}`")

        image_kwargs = dict(image_block_kwargs or {})
        video_kwargs = dict(video_block_kwargs or {})
        content: List[dict[str, Any]] = []

        media_mapping = media or {}
        for media_type in ("image", "audio", "video", "file"):
            media_sources = cls._iter_media_sources(media_mapping.get(media_type))
            for media_source in media_sources:
                formatted_input = await cls._build_media_input_async(
                    media_type,
                    media_source,
                    image_kwargs,
                    video_kwargs,
                )
                if formatted_input:
                    content.append(formatted_input)

        if text not in (None, ""):
            content.append({"type": "text", "text": text})

        return content

    def add_system(self, content: Any) -> None:
        self.add_message("system", content)

    def add_assistant(self, content: Any) -> None:
        self.add_message("assistant", content)

    def add_tool(self, call_id: str, content: Any) -> None:
        self.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def add_reasoning(self, reasoning_content: str, role: str = "assistant") -> None:
        if not isinstance(reasoning_content, str):
            reasoning_content = str(reasoning_content)
        self.append(
            {
                "type": "reasoning",
                "role": role,
                "reasoning_content": reasoning_content,
            }
        )

    def add_assistant_response(
        self, content: Any, reasoning_content: str | None = None
    ) -> None:
        if reasoning_content:
            self.add_reasoning(reasoning_content, role="assistant")
        if content is not None:
            self.add_assistant(content)

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        if not isinstance(metadata, Mapping):
            raise TypeError(f"`metadata` must be Mapping, given `{type(metadata)}`")
        self.metadata.update(self._safe_copy(dict(metadata)))

    def set_response_id(self, response_id: str | None) -> None:
        if response_id is not None and not isinstance(response_id, str):
            response_id = str(response_id)
        self.response_id = response_id

    def to_items(self) -> List[dict[str, Any]]:
        return deepcopy(self._items)

    def set_item_disabled(
        self,
        index: int,
        *,
        disabled: bool = True,
        reason: str | None = None,
    ) -> None:
        item = self._items[index]
        item["disabled"] = bool(disabled)
        if reason is not None:
            item.setdefault("metadata", {})
            item["metadata"]["disabled_reason"] = reason

    def to_chatml(self) -> List[dict[str, Any]]:  # noqa: C901
        messages: List[dict[str, Any]] = []
        for item in self._items:
            if item.get("disabled") is True:
                continue
            item_type = item.get("type")
            if item_type == "turn_marker":
                continue
            if item_type == "reasoning":
                converted_reasoning = self._reasoning_item_to_chatml(item)
                if converted_reasoning is not None:
                    messages.append(converted_reasoning)
                continue
            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                arguments = item.get("arguments")
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments if arguments else "{}",
                                },
                            }
                        ],
                    }
                )
                continue
            if item_type == "function_call_output":
                content = item.get("output")
                if not isinstance(content, str):
                    content = msgspec_dumps(content)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id"),
                        "content": content,
                    }
                )
                continue
            if item_type == "message":
                converted = self._response_message_to_chatml(item)
                if converted is not None:
                    messages.append(converted)
                continue

            role = item.get("role")
            if role in {"user", "assistant", "system", "developer", "tool"}:
                messages.append(deepcopy(item))
                continue
        return messages

    def to_responses_input(self) -> List[dict[str, Any]]:  # noqa: C901
        result: List[dict[str, Any]] = []
        for item in self._items:
            if item.get("disabled") is True:
                continue
            item_type = item.get("type")
            if item_type == "turn_marker":
                continue
            if item_type == "reasoning":
                converted_reasoning = self._reasoning_item_to_responses(item)
                if converted_reasoning is not None:
                    result.append(converted_reasoning)
                continue

            if item_type == "function_call":
                response_item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": item.get("call_id") or item.get("id"),
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "{}",
                }
                if item.get("id") is not None:
                    response_item["id"] = item.get("id")
                if item.get("status") is not None:
                    response_item["status"] = item.get("status")
                result.append(response_item)
                continue

            if item_type == "function_call_output":
                response_item = {
                    "type": "function_call_output",
                    "call_id": item.get("call_id"),
                    "output": self._normalize_tool_output(item.get("output")),
                }
                if item.get("id") is not None:
                    response_item["id"] = item.get("id")
                if item.get("status") is not None:
                    response_item["status"] = item.get("status")
                result.append(response_item)
                continue

            if item_type == "message":
                role = item.get("role")
                if role in {"user", "assistant", "system", "developer"}:
                    result.append(
                        {
                            "type": "message",
                            "role": role,
                            "content": self._normalize_message_content_for_responses(
                                item.get("content")
                            ),
                        }
                    )
                else:
                    result.append(deepcopy(item))
                continue

            role = item.get("role")
            if role == "assistant" and item.get("tool_calls"):
                tool_calls = item.get("tool_calls", [])
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function", {})
                    if not isinstance(function, Mapping):
                        continue
                    result.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments") or "{}",
                        }
                    )
                content = item.get("content")
                if content not in (None, "", []):
                    result.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": self._normalize_message_content_for_responses(
                                content
                            ),
                        }
                    )
                continue

            if role in {"user", "assistant", "system", "developer"}:
                result.append(self._chatml_message_to_response(item))
                continue

            if role == "tool":
                call_id = item.get("tool_call_id")
                output = self._normalize_tool_output(item.get("content"))
                result.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    }
                )
                continue

            if item_type:
                result.append(deepcopy(item))
                continue

            result.append(self._chatml_message_to_response(item))

        return result

    def _to_state(self) -> Mapping[str, Any]:
        return {
            "items": self._safe_copy(self._items),
            "metadata": self._safe_copy(self.metadata),
            "reasoning_content": self.reasoning_content,
            "reasoning_text": self.reasoning_text,
            "response_id": self.response_id,
            "session_id": self.session_id,
            "namespace": self.namespace,
            "turns": self._safe_copy(self._turns),
            "active_turn_index": self._active_turn_index,
        }

    def _hydrate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return

        items = state.get("items")
        if isinstance(items, list):
            self._items = self._safe_copy(items)

        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            self.metadata = self._safe_copy(dict(metadata))
        else:
            self.metadata = {}

        reasoning_content = state.get("reasoning_content")
        self.reasoning_content = (
            reasoning_content if isinstance(reasoning_content, str) else None
        )
        reasoning_text = state.get("reasoning_text")
        self.reasoning_text = (
            reasoning_text if isinstance(reasoning_text, str) else None
        )

        response_id = state.get("response_id")
        self.response_id = response_id if isinstance(response_id, str) else None

        persisted_session_id = state.get("session_id")
        if isinstance(persisted_session_id, str) and persisted_session_id:
            self.session_id = persisted_session_id

        persisted_namespace = state.get("namespace")
        if isinstance(persisted_namespace, str) and persisted_namespace:
            self.namespace = persisted_namespace

        turns = state.get("turns")
        if isinstance(turns, list):
            self._turns = self._safe_copy(turns)
        else:
            self._turns = []

        active_turn_index = state.get("active_turn_index")
        if isinstance(active_turn_index, int):
            self._active_turn_index = active_turn_index
        else:
            self._active_turn_index = None

    def _chatml_message_to_response(self, message: Mapping[str, Any]) -> dict[str, Any]:
        role = message.get("role")
        content = message.get("content")

        tool_calls = message.get("tool_calls")
        if role == "assistant" and tool_calls:
            converted_calls = []
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function", {})
                if not isinstance(function, Mapping):
                    continue
                converted_calls.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            if len(converted_calls) == 1:
                return converted_calls[0]
            return {
                "type": "message",
                "role": "assistant",
                "content": self._normalize_message_content_for_responses(content),
            }

        return {
            "type": "message",
            "role": role,
            "content": self._normalize_message_content_for_responses(content),
        }

    def _normalize_message_content_for_responses(self, content: Any):  # noqa: C901
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, Mapping):
            content = [content]

        if not isinstance(content, list):
            return str(content)

        normalized = []
        for part in content:
            if not isinstance(part, Mapping):
                normalized.append({"type": "input_text", "text": str(part)})
                continue

            part_type = part.get("type")
            if part_type == "text":
                normalized.append({"type": "input_text", "text": part.get("text", "")})
                continue
            if part_type == "output_text":
                normalized.append({"type": "input_text", "text": part.get("text", "")})
                continue

            if part_type == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, Mapping):
                    converted: dict[str, Any] = {
                        "type": "input_image",
                        "image_url": image_url.get("url"),
                    }
                    if image_url.get("detail") is not None:
                        converted["detail"] = image_url.get("detail")
                    normalized.append(converted)
                continue

            if part_type == "file":
                file_item = part.get("file", {})
                if isinstance(file_item, Mapping):
                    converted = {"type": "input_file"}
                    for key in ("file_id", "file_url", "file_data", "filename"):
                        if file_item.get(key) is not None:
                            converted[key] = file_item.get(key)
                    normalized.append(converted)
                continue

            if part_type == "audio_url":
                converted_audio = self._audio_url_to_input_audio(part)
                if converted_audio is not None:
                    normalized.append(converted_audio)
                else:
                    normalized.append(deepcopy(dict(part)))
                continue

            if part_type in {"input_text", "input_image", "input_file", "input_audio"}:
                normalized.append(deepcopy(dict(part)))
                continue

            normalized.append(deepcopy(dict(part)))

        return normalized

    def _normalize_tool_output(self, output: Any):  # noqa: C901
        if output is None:
            return ""

        if isinstance(output, str):
            return output

        if isinstance(output, Mapping):
            output_type = output.get("type")
            if output_type in {
                "input_text",
                "input_image",
                "input_file",
                "input_audio",
            }:
                return [deepcopy(dict(output))]
            if output_type == "text":
                return [{"type": "input_text", "text": output.get("text", "")}]
            if output_type == "image_url":
                image_url = output.get("image_url", {})
                if isinstance(image_url, Mapping):
                    item: dict[str, Any] = {
                        "type": "input_image",
                        "image_url": image_url.get("url"),
                    }
                    if image_url.get("detail") is not None:
                        item["detail"] = image_url.get("detail")
                    return [item]
            if output_type == "file":
                file_item = output.get("file", {})
                if isinstance(file_item, Mapping):
                    item = {"type": "input_file"}
                    for key in ("file_id", "file_url", "file_data", "filename"):
                        if file_item.get(key) is not None:
                            item[key] = file_item.get(key)
                    return [item]
            if output_type == "audio_url":
                converted_audio = self._audio_url_to_input_audio(output)
                if converted_audio is not None:
                    return [converted_audio]
            return msgspec_dumps(output)

        if isinstance(output, list):
            converted_list: list[Any] = []
            for part in output:
                if isinstance(part, Mapping):
                    part_type = part.get("type")
                    if part_type in {
                        "input_text",
                        "input_image",
                        "input_file",
                        "input_audio",
                    }:
                        converted_list.append(deepcopy(dict(part)))
                        continue
                    if part_type == "text":
                        converted_list.append(
                            {
                                "type": "input_text",
                                "text": part.get("text", ""),
                            }
                        )
                        continue
                    if part_type == "image_url":
                        image_url = part.get("image_url", {})
                        if isinstance(image_url, Mapping):
                            item = {
                                "type": "input_image",
                                "image_url": image_url.get("url"),
                            }
                            if image_url.get("detail") is not None:
                                item["detail"] = image_url.get("detail")
                            converted_list.append(item)
                            continue
                    if part_type == "file":
                        file_item = part.get("file", {})
                        if isinstance(file_item, Mapping):
                            item = {"type": "input_file"}
                            for key in ("file_id", "file_url", "file_data", "filename"):
                                if file_item.get(key) is not None:
                                    item[key] = file_item.get(key)
                            converted_list.append(item)
                            continue
                    if part_type == "audio_url":
                        converted_audio = self._audio_url_to_input_audio(part)
                        if converted_audio is not None:
                            converted_list.append(converted_audio)
                            continue
                converted_list.append({"type": "input_text", "text": str(part)})
            return converted_list

        return str(output)

    def _response_part_to_chatml(
        self, part: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        part_type = part.get("type")
        if part_type in ("output_text", "input_text"):
            return {"type": "text", "text": part.get("text", "")}
        if part_type == "input_image":
            image_item: dict[str, Any] = {"url": part.get("image_url")}
            if part.get("detail") is not None:
                image_item["detail"] = part.get("detail")
            return {"type": "image_url", "image_url": image_item}
        if part_type == "input_file":
            file_item = {
                key: part.get(key)
                for key in ("file_id", "file_url", "file_data", "filename")
                if part.get(key) is not None
            }
            return {"type": "file", "file": file_item}
        if part_type == "input_audio":
            input_audio = part.get("input_audio")
            if isinstance(input_audio, Mapping):
                return {"type": "input_audio", "input_audio": dict(input_audio)}
        return None

    def _response_message_to_chatml(
        self, message: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        role = message.get("role")
        content = message.get("content", [])
        if role not in {"user", "assistant", "system", "developer"}:
            return None

        if isinstance(content, str):
            return {"role": role, "content": content}

        if not isinstance(content, list):
            return {"role": role, "content": str(content)}

        chat_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            converted = self._response_part_to_chatml(part)
            if converted is not None:
                chat_content.append(converted)

        if len(chat_content) == 1 and chat_content[0].get("type") == "text":
            return {"role": role, "content": chat_content[0].get("text")}
        if not chat_content:
            return {"role": role, "content": ""}
        return {"role": role, "content": chat_content}

    def _normalize_item(self, item: Mapping[str, Any]) -> List[dict[str, Any]]:
        normalized = deepcopy(dict(item))
        item_type = normalized.get("type")

        if item_type == "reasoning":
            role = normalized.get("role", "assistant")
            reasoning_content = self._extract_reasoning_content(normalized)
            if reasoning_content is None:
                return []
            return [
                {
                    "type": "reasoning",
                    "role": role,
                    "reasoning_content": reasoning_content,
                }
            ]

        role = normalized.get("role")
        if role == "assistant":
            reasoning_content = self._extract_reasoning_content(normalized)
            if reasoning_content:
                for field in ("reasoning_content", "reasoning_text", "think"):
                    normalized.pop(field, None)
                return [
                    {
                        "type": "reasoning",
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                    },
                    normalized,
                ]

        return [normalized]

    def _extract_reasoning_content(self, item: Mapping[str, Any]) -> str | None:
        for field in ("reasoning_content", "reasoning_text", "think"):
            value = item.get(field)
            if isinstance(value, str) and value:
                return value

        if item.get("type") == "reasoning":
            content = item.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                chunks: List[str] = []
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                if chunks:
                    return "".join(chunks)

        return None

    def _reasoning_item_to_chatml(
        self, item: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        reasoning_content = self._extract_reasoning_content(item)
        if reasoning_content is None:
            return None
        role = item.get("role", "assistant")
        return {"role": role, "content": reasoning_content}

    def _reasoning_item_to_responses(
        self, item: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        reasoning_content = self._extract_reasoning_content(item)
        if reasoning_content is None:
            return None
        role = item.get("role", "assistant")
        return {
            "type": "message",
            "role": role,
            "content": self._normalize_message_content_for_responses(reasoning_content),
        }

    def _extract_turn_reasoning(self, turn: Mapping[str, Any]) -> str | None:
        start_item_index = turn.get("start_item_index")
        end_item_index = turn.get("end_item_index")
        if not isinstance(start_item_index, int) or not isinstance(end_item_index, int):
            return None
        if start_item_index < 0 or end_item_index < start_item_index:
            return None

        for item in self._items[start_item_index : end_item_index + 1]:
            if item.get("type") != "reasoning":
                continue
            reasoning_content = self._extract_reasoning_content(item)
            if reasoning_content:
                return reasoning_content
        return None

    @staticmethod
    def _iter_media_sources(media_sources: Any) -> List[Any]:
        if media_sources is None:
            return []
        if isinstance(media_sources, list):
            return media_sources
        return [media_sources]

    @classmethod
    def _build_media_input_sync(
        cls,
        media_type: str,
        media_source: Any,
        image_block_kwargs: Mapping[str, Any],
        video_block_kwargs: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if isinstance(media_source, Mapping):
            return deepcopy(dict(media_source))

        if isinstance(media_source, MediaType):
            return media_source()

        if not isinstance(media_source, str):
            return None

        if media_type == "image":
            return Image(media_source, **image_block_kwargs)()
        if media_type == "audio":
            return Audio(media_source)()
        if media_type == "video":
            is_url = media_source.startswith("http")
            return Video(media_source, force_encode=not is_url, **video_block_kwargs)()
        if media_type == "file":
            return File(media_source)()
        return None

    @classmethod
    async def _build_media_input_async(
        cls,
        media_type: str,
        media_source: Any,
        image_block_kwargs: Mapping[str, Any],
        video_block_kwargs: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if isinstance(media_source, Mapping):
            return deepcopy(dict(media_source))

        if isinstance(media_source, MediaType):
            return await media_source.acall()

        if not isinstance(media_source, str):
            return None

        if media_type == "image":
            return await Image(media_source, **image_block_kwargs).acall()
        if media_type == "audio":
            return await Audio(media_source).acall()
        if media_type == "video":
            is_url = media_source.startswith("http")
            return await Video(
                media_source, force_encode=not is_url, **video_block_kwargs
            ).acall()
        if media_type == "file":
            return await File(media_source).acall()
        return None

    def _audio_url_to_input_audio(
        self, audio_part: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        audio_url = audio_part.get("audio_url")
        if not isinstance(audio_url, Mapping):
            return None

        url = audio_url.get("url")
        if not isinstance(url, str):
            return None

        marker = ";base64,"
        if not url.startswith("data:audio/") or marker not in url:
            return None

        header, audio_data = url.split(marker, 1)
        audio_format = "mp3"
        header_tokens = header.split("/")
        if len(header_tokens) > 1 and header_tokens[1]:
            audio_format = header_tokens[1]
            if ";" in audio_format:
                audio_format = audio_format.split(";", 1)[0]

        return {
            "type": "input_audio",
            "input_audio": {"data": audio_data, "format": audio_format},
        }

    @staticmethod
    def _safe_copy(value: Any) -> Any:
        try:
            return deepcopy(value)
        except Exception:
            return str(value)

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
