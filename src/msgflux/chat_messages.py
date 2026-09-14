"""ChatMessages — container for chat history with serialization support."""

from __future__ import annotations

import json
from copy import deepcopy
from threading import RLock
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Literal, Mapping

from msgflux._private.chat_items import legacy_item_id, new_item_id
from msgflux.core.examples import Example
from msgflux.data.types import Audio, File, Image, MediaType, Video
from msgflux.models.tool_transport import history_adapter, native_item_types
from msgflux.runtime.context import (
    _CURRENT_NAMESPACE,
    _CURRENT_THREAD_ID,
    get_thread_context,
    thread_context,
)
from msgflux.utils.msgspec import msgspec_dumps
from msgflux.utils.time import utc_now_isoformat

if TYPE_CHECKING:
    from msgflux.models.reasoning import ReasoningCodec


class ChatMessages:
    """Provider-neutral interaction timeline with serialization support.

    The class behaves like a mutable list of normalized chat items while also
    carrying thread identity and state serialization helpers used by durable
    agents. Turn lifecycle records live in the timeline; they are not mirrored
    in a separate state structure.
    """

    def __init__(
        self,
        items: Iterable[Mapping[str, Any]] | None = None,
        *,
        thread_id: str | None = None,
        namespace: str | None = None,
    ):
        self._items: List[dict[str, Any]] = []
        self._runtime_lock = RLock()
        self._stream_owner: str | None = None
        self.metadata: dict[str, Any] = {}
        self.thread_id: str | None = (
            thread_id if thread_id is not None else _CURRENT_THREAD_ID.get()
        )
        self.namespace: str | None = (
            namespace if namespace is not None else _CURRENT_NAMESPACE.get()
        )
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
            existing_ids = {
                existing.get("item_id")
                for item_index, existing in enumerate(self._items)
                if item_index != index
            }
            self._assign_item_ids(normalized_items, existing_ids=existing_ids)
            self._items[index] = normalized_items[0]
        else:
            existing_ids = {
                existing.get("item_id")
                for item_index, existing in enumerate(self._items)
                if item_index != index
            }
            self._assign_item_ids(normalized_items, existing_ids=existing_ids)
            self._items[index : index + 1] = normalized_items

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        preview = self._items[-3:]
        return (
            "ChatMessages("
            f"size={len(self._items)}, "
            f"turns={len(self.turns)}, "
            f"thread_id={self.thread_id!r}, "
            f"namespace={self.namespace!r}, "
            f"preview={preview})"
        )

    def append(self, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        normalized_items = self._normalize_item(item)
        self._assign_item_ids(normalized_items)
        self._items.extend(normalized_items)

    def insert(self, index: int, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        normalized_items = self._normalize_item(item)
        self._assign_item_ids(normalized_items)
        self._items[index:index] = normalized_items

    def insert_before_active_turn(self, item: Mapping[str, Any]) -> None:
        if not isinstance(item, Mapping):
            raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
        active_turn = self.get_active_turn()
        if active_turn is None:
            self.append(item)
            return

        normalized_items = self._normalize_item(item)
        self._assign_item_ids(normalized_items)
        start_item_index = active_turn.get("start_item_index")
        if not isinstance(start_item_index, int):
            self.append(item)
            return

        self._items[start_item_index:start_item_index] = normalized_items

    def extend(self, items: Iterable[Mapping[str, Any]]) -> None:
        if isinstance(items, ChatMessages):
            items = items._items
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError(f"`item` must be Mapping, given `{type(item)}`")
            normalized_items = self._normalize_item(item)
            self._assign_item_ids(normalized_items)
            self._items.extend(normalized_items)

    def _assign_item_ids(
        self,
        items: Iterable[dict[str, Any]],
        *,
        existing_ids: set[Any] | None = None,
    ) -> None:
        used_ids = (
            {
                item.get("item_id")
                for item in self._items
                if isinstance(item.get("item_id"), str)
            }
            if existing_ids is None
            else {item_id for item_id in existing_ids if isinstance(item_id, str)}
        )
        for item in items:
            item_id = item.get("item_id")
            if not isinstance(item_id, str) or not item_id or item_id in used_ids:
                item_id = new_item_id()
                item["item_id"] = item_id
            used_ids.add(item_id)

    def copy(self) -> ChatMessages:
        copied = ChatMessages(
            self._items,
            thread_id=self.thread_id,
            namespace=self.namespace,
        )
        copied.metadata = deepcopy(self.metadata)
        return copied

    def __deepcopy__(self, memo: dict[int, Any]) -> ChatMessages:
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def _ensure_stream_available(self) -> None:
        with self._runtime_lock:
            if self._stream_owner is not None:
                raise RuntimeError(
                    "ChatMessages is owned by an unfinished Agent stream "
                    f"(`{self._stream_owner}`). Finish or abort that stream before "
                    "starting another run with the same history."
                )

    def _claim_stream(self, run_id: str) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("A streaming ChatMessages owner requires a run id")
        with self._runtime_lock:
            self._ensure_stream_available()
            self._stream_owner = run_id

    def _release_stream(self, run_id: str) -> None:
        with self._runtime_lock:
            if self._stream_owner == run_id:
                self._stream_owner = None

    thread_context = staticmethod(thread_context)
    get_thread_context = staticmethod(get_thread_context)

    def configure_thread(
        self,
        *,
        thread_id: str | None = None,
        namespace: str | None = None,
    ) -> None:
        resolved_thread_id = (
            thread_id
            if thread_id is not None
            else (
                self.thread_id
                if self.thread_id is not None
                else _CURRENT_THREAD_ID.get()
            )
        )
        self.thread_id = resolved_thread_id
        if namespace is not None:
            self.namespace = namespace
        elif self.namespace is None:
            self.namespace = _CURRENT_NAMESPACE.get()

    def begin_turn(
        self,
        *,
        thread_id: str | None = None,
        namespace: str | None = None,
        turn_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if self.get_active_turn() is not None:
            self.end_turn(event="interrupt")

        self.configure_thread(thread_id=thread_id, namespace=namespace)

        turn_index = len(self.turns)
        turn_identifier = (
            turn_id
            if isinstance(turn_id, str) and turn_id
            else f"turn_{turn_index + 1}"
        )
        turn_event = {
            "type": "turn",
            "event": "start",
            "turn_id": turn_identifier,
            "index": turn_index,
            "thread_id": self.thread_id,
            "namespace": self.namespace,
            "timestamp": utc_now_isoformat(),
        }
        if metadata:
            turn_event["metadata"] = self._safe_copy(dict(metadata))
        self.append(turn_event)

        return turn_identifier

    def end_turn(
        self,
        *,
        event: Literal["pause", "complete", "fail", "interrupt"] = "complete",
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        active_turn = self.get_active_turn()
        if active_turn is None:
            return None
        if event not in {"pause", "complete", "fail", "interrupt"}:
            raise ValueError(f"Unsupported turn event `{event}`")

        turn_event = {
            "type": "turn",
            "event": event,
            "turn_id": active_turn["turn_id"],
            "index": active_turn["index"],
            "thread_id": self.thread_id,
            "namespace": active_turn.get("namespace"),
            "timestamp": utc_now_isoformat(),
        }
        if metadata:
            turn_event["metadata"] = self._safe_copy(dict(metadata))
        self.append(turn_event)
        return deepcopy(self.turns[-1])

    def resume_turn(
        self,
        turn_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self.get_active_turn() is not None:
            raise RuntimeError("Cannot resume a turn while another turn is active")
        matches = [turn for turn in self.turns if turn.get("turn_id") == turn_id]
        if not matches:
            raise ValueError(f"Unknown turn_id `{turn_id}`")
        turn = matches[-1]
        if turn.get("status") not in {"paused", "failed"}:
            raise ValueError(
                f"Turn `{turn_id}` cannot resume from status `{turn.get('status')}`"
            )
        item = {
            "type": "turn",
            "event": "resume",
            "turn_id": turn_id,
            "index": turn["index"],
            "thread_id": self.thread_id,
            "namespace": turn.get("namespace"),
            "timestamp": utc_now_isoformat(),
        }
        if metadata:
            item["metadata"] = self._safe_copy(dict(metadata))
        self.append(item)

    @property
    def turns(self) -> List[dict[str, Any]]:
        return self._derive_turns()

    def get_active_turn(self) -> Mapping[str, Any] | None:
        turns = self._derive_turns()
        if turns and turns[-1].get("status") == "in_progress":
            return turns[-1]
        return None

    def get_active_turn_size(self) -> int:
        active_turn = self.get_active_turn()
        if active_turn is None:
            return 0
        start_item_index = active_turn.get("start_item_index")
        if not isinstance(start_item_index, int):
            return 0
        return len(self._items) - start_item_index

    def latest_completed_turn_boundary(self) -> Mapping[str, Any] | None:
        """Return the latest completed turn that can safely end a compacted prefix."""
        for turn in reversed(self.turns):
            if turn.get("status") != "completed":
                continue
            end_item_index = turn.get("end_item_index")
            if not isinstance(end_item_index, int):
                continue
            boundary = self._items[end_item_index]
            item_id = boundary.get("item_id")
            if isinstance(item_id, str) and item_id:
                return {
                    "item_id": item_id,
                    "item_index": end_item_index,
                    "turn_id": turn.get("turn_id"),
                }
        return None

    def latest_compaction(self) -> Mapping[str, Any] | None:
        """Return the newest append-only compaction operation, if present."""
        for item in reversed(self._items):
            if self._is_compaction_operation(item):
                return deepcopy(item)
        return None

    def through_item(self, item_id: str) -> ChatMessages:
        """Copy the canonical timeline through one inclusive item boundary."""
        index = self._item_index(item_id)
        copied = ChatMessages(
            self._items[: index + 1],
            thread_id=self.thread_id,
            namespace=self.namespace,
        )
        copied.metadata = deepcopy(self.metadata)
        return copied

    def add_compaction(
        self,
        *,
        compacted_through_item_id: str,
        views: Iterable[Mapping[str, Any]],
        reason: str = "threshold",
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Append a complete context-view operation without rewriting history."""
        if (
            not isinstance(compacted_through_item_id, str)
            or not compacted_through_item_id
        ):
            raise ValueError("`compacted_through_item_id` must be a non-empty string")
        boundary_index = self._item_index(compacted_through_item_id)
        boundary = self._items[boundary_index]
        if boundary.get("type") != "turn" or boundary.get("event") != "complete":
            raise ValueError("Compaction must end at a completed turn boundary")

        normalized_views = []
        for view in views:
            if not isinstance(view, Mapping):
                raise TypeError("Compaction views must be mappings")
            view_format = view.get("format")
            items = view.get("items")
            if view_format not in {"messages", "provider"}:
                raise ValueError(
                    "Compaction view format must be `messages` or `provider`"
                )
            if not isinstance(items, list):
                raise TypeError("Compaction view `items` must be a list")
            if view_format == "provider" and (
                not isinstance(view.get("provider"), str)
                or not isinstance(view.get("api_mode"), str)
            ):
                raise ValueError(
                    "Provider compaction views require `provider` and `api_mode`"
                )
            normalized_views.append(self._safe_copy(dict(view)))
        if not normalized_views:
            raise ValueError("Compaction requires at least one complete view")

        previous = self.latest_compaction()
        operation = {
            "type": "compaction",
            "reason": reason,
            "compacted_through_item_id": compacted_through_item_id,
            "views": normalized_views,
            "metadata": self._safe_copy(dict(metadata or {})),
            "timestamp": utc_now_isoformat(),
        }
        if previous is not None and isinstance(previous.get("item_id"), str):
            operation["parent_compaction_id"] = previous["item_id"]
        self.append(operation)
        return deepcopy(self._items[-1])

    def materialize_context(
        self,
        *,
        provider: str | None = None,
        api_mode: str | None = None,
    ) -> ChatMessages:
        """Build the effective model context from the newest compatible view."""
        items = self._materialized_items(provider=provider, api_mode=api_mode)
        materialized = ChatMessages(
            thread_id=self.thread_id,
            namespace=self.namespace,
        )
        # A materialized provider view may contain wire items whose identifiers
        # are meaningful to the provider. Do not normalize or replace them here.
        materialized._items = self._safe_copy(items)
        materialized.metadata = deepcopy(self.metadata)
        return materialized

    def _materialized_items(
        self,
        *,
        provider: str | None,
        api_mode: str | None,
    ) -> List[dict[str, Any]]:
        operations = [
            item
            for item in self._items
            if item.get("type") == "compaction" and isinstance(item.get("views"), list)
        ]
        if not operations:
            return [
                self._safe_copy(item)
                for item in self._items
                if not self._is_compaction_operation(item)
            ]

        operation = operations[-1]
        selected_view = self._select_compaction_view(
            operation,
            provider=provider,
            api_mode=api_mode,
        )
        boundary_id = operation.get("compacted_through_item_id")
        if not isinstance(boundary_id, str):
            raise ValueError("Compaction operation has no valid boundary")
        boundary_index = self._item_index(boundary_id)
        suffix = [
            self._safe_copy(item)
            for item in self._items[boundary_index + 1 :]
            if not self._is_compaction_operation(item)
        ]
        return [
            *self._safe_copy(selected_view["items"]),
            *suffix,
        ]

    @staticmethod
    def _is_compaction_operation(item: Mapping[str, Any]) -> bool:
        return item.get("type") == "compaction" and isinstance(item.get("views"), list)

    def _select_compaction_view(
        self,
        operation: Mapping[str, Any],
        *,
        provider: str | None,
        api_mode: str | None,
    ) -> Mapping[str, Any]:
        views = operation.get("views") or []
        if provider is not None and api_mode is not None:
            for view in views:
                if (
                    isinstance(view, Mapping)
                    and view.get("format") == "provider"
                    and view.get("provider") == provider
                    and view.get("api_mode") == api_mode
                ):
                    return view
        for view in views:
            if isinstance(view, Mapping) and view.get("format") == "messages":
                return view
        target = f"{provider or 'unknown'}/{api_mode or 'unknown'}"
        raise ValueError(
            "The newest compaction has no portable view or provider view "
            f"compatible with `{target}`. Compact again before switching models."
        )

    def _item_index(self, item_id: str) -> int:
        for index, item in enumerate(self._items):
            if item.get("item_id") == item_id:
                return index
        raise ValueError(f"Unknown ChatMessages item_id `{item_id}`")

    def fork(self, *, upto_turn: int | None = None) -> ChatMessages:
        if upto_turn is None:
            return self.copy()

        if not isinstance(upto_turn, int):
            raise TypeError(
                f"`upto_turn` must be int or None, given `{type(upto_turn)}`"
            )

        if upto_turn < 0:
            forked = ChatMessages(
                thread_id=self.thread_id,
                namespace=self.namespace,
            )
            forked.metadata = deepcopy(self.metadata)
            return forked

        turns = self.turns
        if upto_turn >= len(turns):
            return self.copy()

        selected_turn = turns[upto_turn]
        end_item_index = selected_turn.get("end_item_index")
        if not isinstance(end_item_index, int):
            end_item_index = len(self._items) - 1

        forked = ChatMessages(
            self._items[: end_item_index + 1],
            thread_id=self.thread_id,
            namespace=self.namespace,
        )
        forked.metadata = deepcopy(self.metadata)
        return forked

    def to_examples(self) -> List[Example]:
        """Split completed turns into provider-neutral input/output trajectories."""
        examples: List[Example] = []

        for turn in self.turns:
            if turn.get("status") != "completed":
                continue
            start = int(turn["start_item_index"]) + 1
            end = int(turn["end_item_index"])
            trajectory = self._items[start:end]
            example_trajectory = [
                {
                    key: self._safe_copy(value)
                    for key, value in item.items()
                    if key not in {"item_id", "metadata"}
                }
                for item in trajectory
                if not self._is_compaction_operation(item)
            ]
            split = next(
                (
                    index
                    for index, item in enumerate(example_trajectory)
                    if self._is_assistant_trajectory_item(item)
                ),
                len(example_trajectory),
            )
            examples.append(
                Example(
                    inputs={"trajectory": example_trajectory[:split]},
                    labels={"trajectory": example_trajectory[split:]},
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

    def close_interrupted_tool_calls(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        open_call_ids: list[str] = []
        native_calls = {}
        closed_call_ids: set[str] = set()
        for item in self._items:
            if item.get("type") in {"function_call", *native_item_types()}:
                call_id = item.get("call_id") or item.get("id")
                if isinstance(call_id, str) and call_id:
                    open_call_ids.append(call_id)
                    if item.get("type") in native_item_types():
                        native_calls[call_id] = item
                continue
            if item.get("type") in {
                "function_call_output",
                *native_item_types(output=True),
            }:
                call_id = item.get("call_id")
                if isinstance(call_id, str) and call_id:
                    closed_call_ids.add(call_id)

        missing_call_ids = [
            call_id for call_id in open_call_ids if call_id not in closed_call_ids
        ]
        for call_id in missing_call_ids:
            if call_id in native_calls:
                item = native_calls[call_id]
                self.append(
                    history_adapter(item).interrupted(
                        item,
                        reason or "Tool call interrupted; effects may be unconfirmed.",
                    )
                )
                continue
            self.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": self._interrupted_tool_call_output(reason),
                    "status": "interrupted",
                }
            )
        return len(missing_call_ids)

    @staticmethod
    def _interrupted_tool_call_output(reason: str | None = None) -> dict[str, str]:
        output = {
            "status": "interrupted",
            "reason": "user_requested_stop",
            "message": "Tool call interrupted by user interrupt request.",
        }
        if reason:
            output["details"] = reason
        return output

    def add_reasoning(
        self,
        text: str | None = None,
        *,
        summary: str | None = None,
        provider_state: Any = None,
        provider: str | None = None,
        role: str = "assistant",
    ) -> None:
        """Append normalized reasoning and optional opaque provider state."""
        item: dict[str, Any] = {"type": "reasoning", "role": role}
        if text is not None:
            item["text"] = str(text)
        if summary is not None:
            item["summary"] = str(summary)
        if provider_state is not None:
            item["provider_state"] = {
                "provider": provider,
                "data": self._safe_copy(provider_state),
            }
        if len(item) == 2:
            return
        self.append(item)

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

    def get_loaded_tools(self, catalog_id: str) -> set[str]:
        """Return deferred tools loaded for this thread and catalog."""
        catalogs = self.metadata.get("tool_catalogs")
        if not isinstance(catalogs, Mapping):
            return set()
        state = catalogs.get(catalog_id)
        if not isinstance(state, Mapping):
            return set()
        loaded = state.get("loaded_tools")
        if not isinstance(loaded, list):
            return set()
        return {name for name in loaded if isinstance(name, str) and name}

    def load_tools(self, catalog_id: str, tool_names: Iterable[str]) -> list[str]:
        """Mark deferred tools as loaded in this thread's durable state."""
        names = {name for name in tool_names if isinstance(name, str) and name}
        if not names:
            return []
        catalogs = self.metadata.setdefault("tool_catalogs", {})
        if not isinstance(catalogs, dict):
            raise TypeError("`metadata['tool_catalogs']` must be a mapping")
        state = catalogs.setdefault(catalog_id, {})
        if not isinstance(state, dict):
            raise TypeError(f"Tool catalog state for `{catalog_id}` must be a mapping")
        loaded = self.get_loaded_tools(catalog_id)
        loaded.update(names)
        state["loaded_tools"] = sorted(loaded)
        return sorted(names)

    def to_items(self) -> List[dict[str, Any]]:
        return deepcopy(self._items)

    def pprint(
        self,
        *,
        provider: str | None = None,
        api_mode: str | None = None,
        show_turns: bool = False,
        show_compaction: bool = False,
        show_reasoning: bool = True,
        show_system: bool = True,
        max_content_chars: int | None = 2000,
        max_tool_args_chars: int | None = 1000,
        max_tool_output_chars: int | None = 2000,
    ) -> str:
        """Return an elegant plain-text view of the conversation.

        Blocks are separated by a blank line and prefixed with a header
        so each kind is visually distinct::

            [SYSTEM] / [DEVELOPER] / [USER] / [ASSISTANT] / [REASONING]
            [TOOL CALL <name> id=<call_id>]
            [TOOL RESULT id=<call_id> status=<status>]

        Turn and compaction bookkeeping items are skipped unless
        ``show_turns`` / ``show_compaction`` is enabled. Multimodal and
        binary parts are replaced by short placeholders (``[image: ...]``,
        ``[file: ...]``) instead of dumping raw payloads.

        ``provider`` / ``api_mode`` are optional and only matter when the
        history contains a provider-only compaction view (no portable
        ``messages`` view). By default the portable view is used, so there
        is nothing to detect or pass.
        """
        blocks: List[str] = []
        for stored_item in self._materialized_items(
            provider=provider, api_mode=api_mode
        ):
            item = stored_item
            adapter = history_adapter(item)
            if adapter is not None:
                item = adapter.project_history(item)
            block = self._pprint_item(
                item,
                show_turns=show_turns,
                show_compaction=show_compaction,
                show_reasoning=show_reasoning,
                show_system=show_system,
                max_content_chars=max_content_chars,
                max_tool_args_chars=max_tool_args_chars,
                max_tool_output_chars=max_tool_output_chars,
            )
            if block:
                blocks.append(block)
        return "\n\n".join(blocks)

    def _pprint_item(  # noqa: C901
        self,
        item: Mapping[str, Any],
        *,
        show_turns: bool,
        show_compaction: bool,
        show_reasoning: bool,
        show_system: bool,
        max_content_chars: int | None,
        max_tool_args_chars: int | None,
        max_tool_output_chars: int | None,
    ) -> str | None:
        item_type = item.get("type")

        if item_type == "turn":
            if not show_turns:
                return None
            turn_id = item.get("turn_id", "?")
            event = item.get("event", "?")
            return f"--- turn {turn_id} ({event}) ---"

        if item_type == "compaction":
            if not show_compaction:
                return None
            reason = item.get("reason", "?")
            return f"--- compaction ({reason}) ---"

        if item_type == "reasoning":
            if not show_reasoning:
                return None
            text = self._extract_reasoning_content(item)
            if text is None and isinstance(item.get("summary"), str):
                text = item["summary"]
            return f"[REASONING]\n{self._truncate(text or '', max_content_chars)}"

        if item_type == "function_call":
            name = item.get("name") or "unknown_tool"
            call_id = item.get("call_id") or item.get("id") or "?"
            args = self._pprint_jsonish(item.get("arguments"))
            return (
                f"[TOOL CALL {name} id={call_id}]\n"
                f"{self._truncate(args, max_tool_args_chars)}"
            )

        if item_type == "function_call_output":
            call_id = item.get("call_id") or "?"
            status = item.get("status")
            header = f"[TOOL RESULT id={call_id}"
            if isinstance(status, str) and status:
                header += f" status={status}"
            header += "]"
            output = self._pprint_tool_output(item.get("output"))
            return f"{header}\n{self._truncate(output, max_tool_output_chars)}"

        if item_type in {"tool_search_call", "tool_search_output"}:
            return self._pprint_tool_search_item(
                item, max_tool_args_chars, max_tool_output_chars
            )

        if item_type == "message":
            role = item.get("role", "?")
            if role == "system" and not show_system:
                return None
            header = {
                "system": "[SYSTEM]",
                "developer": "[DEVELOPER]",
                "user": "[USER]",
                "assistant": "[ASSISTANT]",
            }.get(role, f"[{str(role).upper()}]")
            content = self._pprint_content(item.get("content"))
            return f"{header}\n{self._truncate(content, max_content_chars)}"

        # Legacy / wire shapes (ChatML roles).
        role = item.get("role")
        if role == "system":
            if not show_system:
                return None
            content = self._pprint_content(item.get("content"))
            return f"[SYSTEM]\n{self._truncate(content, max_content_chars)}"
        if role in {"user", "developer"}:
            header = "[USER]" if role == "user" else "[DEVELOPER]"
            content = self._pprint_content(item.get("content"))
            return f"{header}\n{self._truncate(content, max_content_chars)}"
        if role == "assistant":
            return self._pprint_legacy_assistant(
                item,
                show_reasoning=show_reasoning,
                max_content_chars=max_content_chars,
                max_tool_args_chars=max_tool_args_chars,
            )
        if role == "tool":
            call_id = item.get("tool_call_id") or item.get("call_id") or "?"
            output = self._pprint_tool_output(item.get("content"))
            header = f"[TOOL RESULT id={call_id}]"
            return f"{header}\n{self._truncate(output, max_tool_output_chars)}"

        return None

    def _pprint_legacy_assistant(
        self,
        item: Mapping[str, Any],
        *,
        show_reasoning: bool,
        max_content_chars: int | None,
        max_tool_args_chars: int | None,
    ) -> str | None:
        blocks: List[str] = []
        if show_reasoning:
            reasoning = self._extract_reasoning_content(item)
            if reasoning:
                blocks.append(
                    f"[REASONING]\n{self._truncate(reasoning, max_content_chars)}"
                )
        content = self._pprint_content(item.get("content"))
        if content:
            blocks.append(f"[ASSISTANT]\n{self._truncate(content, max_content_chars)}")
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                name = call.get("id") or "?"
                args = "{}"
                if isinstance(function, Mapping):
                    name = function.get("name") or name
                    args = self._pprint_jsonish(function.get("arguments"))
                else:
                    args = self._pprint_jsonish(call.get("arguments"))
                call_id = call.get("id") or "?"
                blocks.append(
                    f"[TOOL CALL {name} id={call_id}]\n"
                    f"{self._truncate(args, max_tool_args_chars)}"
                )
        if not blocks:
            blocks.append("[ASSISTANT]\n")
        return "\n\n".join(blocks)

    def _pprint_tool_search_item(
        self,
        item: Mapping[str, Any],
        max_args_chars: int | None,
        max_output_chars: int | None,
    ) -> str:
        if item.get("type") == "tool_search_call":
            call_id = item.get("id") or item.get("call_id") or "?"
            args = self._pprint_jsonish(item.get("arguments"))
            return (
                f"[TOOL CALL tool_search id={call_id}]\n"
                f"{self._truncate(args, max_args_chars)}"
            )
        call_id = item.get("tool_search_call_id") or item.get("id") or "?"
        output = self._pprint_tool_output(item.get("tools", item.get("output", "")))
        return f"[TOOL RESULT id={call_id}]\n{self._truncate(output, max_output_chars)}"

    @staticmethod
    def _truncate(text: str, limit: int | None) -> str:
        if limit is None or limit < 0 or len(text) <= limit:
            return text
        return f"{text[:limit]}… [+{len(text) - limit} chars]"

    def _pprint_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            content = [content]
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                rendered = self._pprint_content_part(part)
                if rendered:
                    parts.append(rendered)
            return "\n".join(parts)
        return str(content)

    def _pprint_content_part(self, part: Any) -> str:
        if not isinstance(part, Mapping):
            return str(part)
        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            return str(part.get("text", ""))
        if part_type in {"image_url", "input_image"}:
            url = part.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url", "?")
            return f"[image: {url}]"
        if part_type in {"file", "input_file"}:
            file_info = part.get("file", part)
            name = (
                file_info.get("filename", "?")
                if isinstance(file_info, Mapping)
                else "?"
            )
            return f"[file: {name}]"
        if part_type in {"audio_url", "input_audio"}:
            return "[audio]"
        if part_type in {"video_url", "input_video"}:
            url = part.get("video_url", "?")
            return f"[video: {url}]"
        try:
            return msgspec_dumps(part)
        except Exception:
            return str(part)

    def _pprint_tool_output(self, output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        # Dict/list outputs (incl. multimodal) are JSON-encoded; binary
        # parts would already have been replaced by placeholders above
        # only for message content, so keep tool payloads faithful.
        try:
            return msgspec_dumps(output)
        except Exception:
            return str(output)

    @staticmethod
    def _pprint_jsonish(value: Any) -> str:
        if value is None:
            return "{}"
        if isinstance(value, str):
            text = value.strip() or "{}"
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                return text
            try:
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except (ValueError, TypeError):
                return text
        if isinstance(value, Mapping):
            try:
                return json.dumps(dict(value), indent=2, ensure_ascii=False)
            except (ValueError, TypeError):
                return str(value)
        return str(value)

    def to_chatml(  # noqa: C901
        self,
        *,
        provider: str | None = None,
        api_mode: str = "chat_completions",
        reasoning_codec: ReasoningCodec | None = None,
    ) -> List[dict[str, Any]]:
        messages: List[dict[str, Any]] = []
        pending_reasoning: list[Mapping[str, Any]] = []
        for stored_item in self._materialized_items(
            provider=provider, api_mode=api_mode
        ):
            item = stored_item
            item_type = item.get("type")
            adapter = history_adapter(item)
            if adapter is not None:
                item = adapter.project_history(item)
                item_type = item["type"]
            if item_type == "turn":
                continue
            if item_type == "reasoning":
                pending_reasoning.append(item)
                continue
            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                arguments = item.get("arguments")
                converted_tool_call = self._provider_state_mapping(
                    item,
                    provider,
                    api_mode,
                )
                converted_tool_call.update(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments if arguments else "{}",
                        },
                    }
                )
                converted_call = {
                    "role": "assistant",
                    "tool_calls": [converted_tool_call],
                }
                if pending_reasoning:
                    self._apply_reasoning_items_to_chatml(
                        converted_call,
                        pending_reasoning,
                        provider,
                        api_mode,
                        reasoning_codec,
                    )
                    pending_reasoning = []
                if (
                    messages
                    and messages[-1].get("role") == "assistant"
                    and isinstance(messages[-1].get("tool_calls"), list)
                    and "reasoning_details" not in converted_call
                    and "reasoning_content" not in converted_call
                ):
                    messages[-1]["tool_calls"].extend(converted_call["tool_calls"])
                else:
                    messages.append(converted_call)
                continue
            if item_type == "function_call_output":
                if pending_reasoning:
                    converted_reasoning = self._reasoning_items_to_chatml(
                        pending_reasoning,
                        provider=provider,
                        api_mode=api_mode,
                        reasoning_codec=reasoning_codec,
                    )
                    if converted_reasoning is not None:
                        messages.append(converted_reasoning)
                    pending_reasoning = []
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
                converted = self._response_message_to_chatml(
                    item,
                    provider=provider,
                    api_mode=api_mode,
                )
                if converted is not None:
                    if pending_reasoning and converted.get("role") == "assistant":
                        self._apply_reasoning_items_to_chatml(
                            converted,
                            pending_reasoning,
                            provider,
                            api_mode,
                            reasoning_codec,
                        )
                        pending_reasoning = []
                    elif pending_reasoning:
                        converted_reasoning = self._reasoning_items_to_chatml(
                            pending_reasoning,
                            provider=provider,
                            api_mode=api_mode,
                            reasoning_codec=reasoning_codec,
                        )
                        if converted_reasoning is not None:
                            messages.append(converted_reasoning)
                        pending_reasoning = []
                    messages.append(converted)
                continue

            role = item.get("role")
            if role in {"user", "assistant", "system", "developer", "tool"}:
                converted = deepcopy(item)
                if pending_reasoning and role == "assistant":
                    self._apply_reasoning_items_to_chatml(
                        converted,
                        pending_reasoning,
                        provider,
                        api_mode,
                        reasoning_codec,
                    )
                    pending_reasoning = []
                elif pending_reasoning:
                    converted_reasoning = self._reasoning_items_to_chatml(
                        pending_reasoning,
                        provider=provider,
                        api_mode=api_mode,
                        reasoning_codec=reasoning_codec,
                    )
                    if converted_reasoning is not None:
                        messages.append(converted_reasoning)
                    pending_reasoning = []
                messages.append(converted)
                continue
        if pending_reasoning:
            converted_reasoning = self._reasoning_items_to_chatml(
                pending_reasoning,
                provider=provider,
                api_mode=api_mode,
                reasoning_codec=reasoning_codec,
            )
            if converted_reasoning is not None:
                messages.append(converted_reasoning)
        return messages

    def to_responses_input(  # noqa: C901
        self,
        *,
        provider: str = "openai",
        api_mode: str = "responses",
        reasoning_codec: ReasoningCodec | None = None,
        native_tools: bool = True,
    ) -> List[dict[str, Any]]:
        result: List[dict[str, Any]] = []
        for stored_item in self._materialized_items(
            provider=provider, api_mode=api_mode
        ):
            item = stored_item
            adapter = history_adapter(item)
            if adapter is not None:
                if (
                    native_tools
                    and provider == adapter.provider
                    and api_mode == adapter.api_mode
                ):
                    result.append(
                        {
                            key: deepcopy(value)
                            for key, value in item.items()
                            if key not in {"item_id", "metadata"}
                        }
                    )
                    continue
                item = adapter.project_history(item)
            item_type = item.get("type")
            if item_type == "turn":
                continue
            if item_type == "reasoning":
                converted_reasoning = self._reasoning_item_to_responses(
                    item,
                    provider=provider,
                    api_mode=api_mode,
                    reasoning_codec=reasoning_codec,
                )
                if converted_reasoning is not None:
                    result.append(converted_reasoning)
                continue

            if item_type in {"tool_search_call", "tool_search_output"}:
                if self._provider_state_matches(item, provider, api_mode):
                    result.append(
                        self._provider_state_mapping(item, provider, api_mode)
                    )
                continue

            if item_type == "function_call":
                response_item = self._provider_state_mapping(
                    item,
                    provider,
                    api_mode,
                )
                response_item.update(
                    {
                        "type": "function_call",
                        "call_id": item.get("call_id") or item.get("id"),
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    }
                )
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
                status = item.get("status")
                if status == "interrupted":
                    status = "incomplete"
                if status is not None:
                    response_item["status"] = status
                result.append(response_item)
                continue

            if item_type == "message":
                role = item.get("role")
                if role in {"user", "assistant", "system", "developer"}:
                    native_state = self._provider_state_mapping(
                        item, provider, api_mode
                    )
                    has_native_state = self._provider_state_matches(
                        item, provider, api_mode
                    )
                    native_state.update(
                        {
                            "type": "message",
                            "role": role,
                            "content": (
                                deepcopy(item.get("content"))
                                if has_native_state
                                else self._normalize_message_content_for_responses(
                                    item.get("content")
                                )
                            ),
                        }
                    )
                    if item.get("phase") is not None:
                        native_state["phase"] = item.get("phase")
                    result.append(native_state)
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
                converted = deepcopy(item)
                converted.pop("item_id", None)
                converted.pop("metadata", None)
                result.append(converted)
                continue

            result.append(self._chatml_message_to_response(item))

        return result

    def _to_state(self) -> Mapping[str, Any]:
        return {
            "items": self._safe_copy(self._items),
            "metadata": self._safe_copy(self.metadata),
            "thread_id": self.thread_id,
            "namespace": self.namespace,
        }

    def _hydrate_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return

        persisted_thread_id = state.get("thread_id")
        if isinstance(persisted_thread_id, str) and persisted_thread_id:
            self.thread_id = persisted_thread_id

        persisted_namespace = state.get("namespace")
        if isinstance(persisted_namespace, str) and persisted_namespace:
            self.namespace = persisted_namespace

        items = state.get("items")
        if isinstance(items, list):
            restored_items = self._safe_copy(items)
            seen_ids: set[str] = set()
            for index, item in enumerate(restored_items):
                if not isinstance(item, dict):
                    continue
                item_id = item.get("item_id")
                if not isinstance(item_id, str) or not item_id:
                    item_id = legacy_item_id(
                        namespace=self.namespace,
                        thread_id=self.thread_id,
                        index=index,
                        item=item,
                    )
                    item["item_id"] = item_id
                if item_id in seen_ids:
                    raise ValueError(f"Duplicate ChatMessages item_id `{item_id}`.")
                seen_ids.add(item_id)
            self._items = restored_items

        metadata = state.get("metadata")
        if isinstance(metadata, Mapping):
            self.metadata = self._safe_copy(dict(metadata))
        else:
            self.metadata = {}

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
        if part_type in {"text", "image_url", "video_url", "file", "audio_url"}:
            return deepcopy(dict(part))
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
        self,
        message: Mapping[str, Any],
        *,
        provider: str | None = None,
        api_mode: str = "chat_completions",
    ) -> dict[str, Any] | None:
        role = message.get("role")
        content = message.get("content", [])
        if role not in {"user", "assistant", "system", "developer"}:
            return None

        if isinstance(content, str):
            converted = {"role": role, "content": content}
            return self._merge_provider_state(converted, message, provider, api_mode)

        if not isinstance(content, list):
            converted = {"role": role, "content": str(content)}
            return self._merge_provider_state(converted, message, provider, api_mode)

        chat_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            converted = self._response_part_to_chatml(part)
            if converted is not None:
                chat_content.append(converted)

        if len(chat_content) == 1 and chat_content[0].get("type") == "text":
            converted = {"role": role, "content": chat_content[0].get("text")}
            return self._merge_provider_state(converted, message, provider, api_mode)
        if not chat_content:
            converted = {"role": role, "content": ""}
            return self._merge_provider_state(converted, message, provider, api_mode)
        converted = {"role": role, "content": chat_content}
        return self._merge_provider_state(converted, message, provider, api_mode)

    def _normalize_item(  # noqa: C901
        self, item: Mapping[str, Any]
    ) -> List[dict[str, Any]]:
        normalized = deepcopy(dict(item))
        item_type = normalized.get("type")

        if item_type == "reasoning":
            text = self._extract_reasoning_content(normalized)
            summary = normalized.get("summary")
            provider_state = normalized.get("provider_state")
            if text is None and summary is None and provider_state is None:
                return []
            for field in ("reasoning_content", "reasoning_text", "think", "content"):
                normalized.pop(field, None)
            normalized["role"] = normalized.get("role", "assistant")
            if text is not None:
                normalized["text"] = text
            return [normalized]

        if item_type in {
            "turn",
            "message",
            "function_call",
            "function_call_output",
            "tool_search_call",
            "tool_search_output",
        }:
            return [normalized]

        role = normalized.get("role")
        if role == "assistant":
            item_attrs = {
                key: normalized[key] for key in ("metadata",) if key in normalized
            }
            reasoning_content = self._extract_reasoning_content(normalized)
            tool_calls = normalized.pop("tool_calls", None)
            if reasoning_content:
                for field in ("reasoning_content", "reasoning_text", "think"):
                    normalized.pop(field, None)
            result: list[dict[str, Any]] = []
            if reasoning_content:
                result.append(
                    {
                        "type": "reasoning",
                        "role": "assistant",
                        "text": reasoning_content,
                        **item_attrs,
                    }
                )
            content = normalized.pop("content", None)
            normalized.pop("role", None)
            if content not in (None, "", []):
                result.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                        **normalized,
                    }
                )
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    result.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments") or "{}",
                            **item_attrs,
                        }
                    )
            return result or [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "" if content is None else content,
                    **normalized,
                }
            ]

        if role == "tool":
            return [
                {
                    "type": "function_call_output",
                    "call_id": normalized.get("tool_call_id"),
                    "output": normalized.get("content"),
                    **{
                        key: normalized[key]
                        for key in ("metadata",)
                        if key in normalized
                    },
                }
            ]

        if role in {"user", "system", "developer"}:
            normalized.pop("role", None)
            content = normalized.pop("content", None)
            return [
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                    **normalized,
                }
            ]

        return [normalized]

    def _extract_reasoning_content(self, item: Mapping[str, Any]) -> str | None:
        for field in ("text", "reasoning_content", "reasoning_text", "think"):
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

    def _derive_turns(self) -> List[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        status_by_event = {
            "pause": "paused",
            "complete": "completed",
            "fail": "failed",
            "interrupt": "interrupted",
        }
        for item_index, item in enumerate(self._items):
            if item.get("type") != "turn":
                continue
            turn_id = item.get("turn_id")
            event = item.get("event")
            if not isinstance(turn_id, str) or not isinstance(event, str):
                continue
            if event == "start":
                turn = {
                    "turn_id": turn_id,
                    "index": item.get("index", len(turns)),
                    "thread_id": item.get("thread_id", self.thread_id),
                    "namespace": item.get("namespace", self.namespace),
                    "started_at": item.get("timestamp"),
                    "ended_at": None,
                    "status": "in_progress",
                    "start_item_index": item_index,
                    "end_item_index": None,
                    "events": [self._safe_copy(item)],
                }
                turns.append(turn)
                by_id[turn_id] = turn
                continue
            turn = by_id.get(turn_id)
            if turn is None:
                continue
            turn["events"].append(self._safe_copy(item))
            if event == "resume":
                turn["status"] = "in_progress"
                turn["ended_at"] = None
                turn["end_item_index"] = None
            elif event in status_by_event:
                turn["status"] = status_by_event[event]
                turn["ended_at"] = item.get("timestamp")
                turn["end_item_index"] = item_index
        return deepcopy(turns)

    @staticmethod
    def _is_assistant_trajectory_item(item: Mapping[str, Any]) -> bool:
        return item.get("role") == "assistant" or item.get("type") in {
            "reasoning",
            "function_call",
        }

    def _reasoning_items_to_chatml(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        provider: str | None = None,
        api_mode: str = "chat_completions",
        reasoning_codec: ReasoningCodec | None = None,
    ) -> dict[str, Any] | None:
        reasoning_items = list(items)
        if not reasoning_items:
            return None
        role = reasoning_items[-1].get("role", "assistant")
        message: dict[str, Any] = {"role": role}
        self._apply_reasoning_items_to_chatml(
            message,
            reasoning_items,
            provider,
            api_mode,
            reasoning_codec,
        )
        return message if len(message) > 1 else None

    def _apply_reasoning_items_to_chatml(
        self,
        message: dict[str, Any],
        items: Iterable[Mapping[str, Any]],
        provider: str | None,
        api_mode: str,
        reasoning_codec: ReasoningCodec | None,
    ) -> None:
        items = list(items)
        if reasoning_codec is not None and provider is not None:
            message.update(
                reasoning_codec.encode_chat_message(
                    items,
                    provider=provider,
                    api_mode=api_mode,
                )
            )
            return

        reasoning_chunks: list[str] = []
        for item in items:
            reasoning_content = self._extract_reasoning_content(item)
            if reasoning_content is None and isinstance(item.get("summary"), str):
                reasoning_content = item["summary"]
            if reasoning_content is not None:
                reasoning_chunks.append(reasoning_content)
        if reasoning_chunks:
            message["reasoning_content"] = "".join(reasoning_chunks)

    def _provider_state_mapping(
        self,
        item: Mapping[str, Any],
        provider: str | None,
        api_mode: str | None = None,
    ) -> dict[str, Any]:
        provider_state = item.get("provider_state")
        if (
            provider is None
            or not isinstance(provider_state, Mapping)
            or provider_state.get("provider") != provider
            or (
                provider_state.get("api_mode") is not None
                and provider_state.get("api_mode") != api_mode
            )
        ):
            return {}
        data = provider_state.get("data")
        if not isinstance(data, Mapping):
            return {}
        return self._safe_copy(dict(data))

    @staticmethod
    def _provider_state_matches(
        item: Mapping[str, Any],
        provider: str | None,
        api_mode: str | None,
    ) -> bool:
        provider_state = item.get("provider_state")
        return bool(
            provider is not None
            and isinstance(provider_state, Mapping)
            and provider_state.get("provider") == provider
            and provider_state.get("api_mode") in {None, api_mode}
            and isinstance(provider_state.get("data"), Mapping)
        )

    def _merge_provider_state(
        self,
        converted: dict[str, Any],
        item: Mapping[str, Any],
        provider: str | None,
        api_mode: str | None = None,
    ) -> dict[str, Any]:
        merged = self._provider_state_mapping(item, provider, api_mode)
        merged.update(converted)
        return merged

    def _reasoning_item_to_responses(
        self,
        item: Mapping[str, Any],
        *,
        provider: str,
        api_mode: str,
        reasoning_codec: ReasoningCodec | None,
    ) -> dict[str, Any] | None:
        if reasoning_codec is not None:
            return reasoning_codec.encode_responses_item(
                item,
                provider=provider,
                api_mode=api_mode,
            )

        # Backwards-compatible standalone conversion. Model providers always
        # supply a codec and therefore use their explicit replay contract.
        reasoning_content = self._extract_reasoning_content(item)
        provider_state = item.get("provider_state")
        if isinstance(provider_state, Mapping):
            if provider_state.get("provider") == provider and provider_state.get(
                "api_mode"
            ) in {None, api_mode}:
                data = provider_state.get("data")
                if isinstance(data, Mapping):
                    response_item = self._safe_copy(dict(data))
                    if "summary" not in response_item:
                        summary = item.get("summary")
                        if isinstance(summary, str) and summary:
                            response_item["summary"] = [
                                {"type": "summary_text", "text": summary}
                            ]
                    return response_item
        if reasoning_content is None:
            summary = item.get("summary")
            reasoning_content = summary if isinstance(summary, str) else None
        if reasoning_content is None:
            return None
        role = item.get("role", "assistant")
        return {
            "type": "message",
            "role": role,
            "content": self._normalize_message_content_for_responses(reasoning_content),
        }

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
