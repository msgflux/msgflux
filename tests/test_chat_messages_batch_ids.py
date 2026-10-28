from copy import deepcopy
from typing import Any

import pytest

from msgflux.chat_messages import ChatMessages


def _message(content: str, *, item_id: Any = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "user",
        "content": content,
    }
    if item_id is not None:
        item["item_id"] = item_id
    return item


def _item_ids(messages: ChatMessages) -> list[Any]:
    return [item["item_id"] for item in messages]


def _assert_unique_non_empty_ids(messages: ChatMessages) -> None:
    item_ids = _item_ids(messages)
    assert all(isinstance(item_id, str) and item_id for item_id in item_ids)
    assert len(set(item_ids)) == len(item_ids)


def test_constructor_reuses_explicit_ids_and_reassigns_duplicates():
    messages = ChatMessages(
        [
            _message("first", item_id="itm_shared"),
            _message("duplicate", item_id="itm_shared"),
            _message("missing"),
            _message("empty", item_id=""),
            _message("wrong-type", item_id=123),
        ]
    )

    assert _item_ids(messages)[0] == "itm_shared"
    assert _item_ids(messages)[1] != "itm_shared"
    _assert_unique_non_empty_ids(messages)


def test_extend_reuses_one_id_set_for_the_whole_batch():
    messages = ChatMessages([_message("existing", item_id="itm_existing")])
    messages.extend(
        [
            _message("explicit", item_id="itm_explicit"),
            _message("collides-with-existing", item_id="itm_existing"),
            _message("collides-with-batch", item_id="itm_explicit"),
            _message("generated"),
        ]
    )

    assert messages[0]["item_id"] == "itm_existing"
    assert messages[1]["item_id"] == "itm_explicit"
    assert messages[2]["item_id"] != "itm_existing"
    assert messages[3]["item_id"] != "itm_explicit"
    _assert_unique_non_empty_ids(messages)


def test_extend_from_chat_messages_preserves_ids_without_sharing_items():
    source = ChatMessages(
        [
            _message("first", item_id="itm_first"),
            _message("second", item_id="itm_second"),
        ]
    )
    target = ChatMessages()

    target.extend(source)

    assert _item_ids(target) == _item_ids(source)
    assert target[0] is not source[0]
    target[0]["content"] = "changed"
    assert source[0]["content"] == "first"


def test_copy_preserves_ids_and_deep_copies_items_and_metadata():
    source = ChatMessages(
        [
            _message("first", item_id="itm_first"),
            _message("second", item_id="itm_second"),
        ]
    )
    source.metadata["branch"] = {"items": ["original"]}

    copied = source.copy()

    assert _item_ids(copied) == _item_ids(source)
    assert _item_ids(deepcopy(source)) == _item_ids(source)
    assert copied[0] is not source[0]
    copied[0]["content"] = "changed"
    copied.metadata["branch"]["items"].append("copied")
    assert source[0]["content"] == "first"
    assert source.metadata["branch"]["items"] == ["original"]


def test_copy_preserves_legacy_hydrated_ids():
    state = {
        "items": [{"type": "message", "role": "user", "content": "legacy"}],
        "thread_id": "thread_1",
        "namespace": "agent:test",
    }
    messages = ChatMessages()
    messages._hydrate_state(state)

    copied = messages.copy()

    assert _item_ids(copied) == _item_ids(messages)
    assert len(set(_item_ids(copied))) == 1


def test_append_and_insert_preserve_explicit_ids_without_duplicates():
    messages = ChatMessages([_message("existing", item_id="itm_existing")])

    messages.append(_message("appended", item_id="itm_appended"))
    messages.insert(0, _message("inserted", item_id="itm_appended"))

    assert messages[0]["item_id"] != "itm_appended"
    assert messages[1]["item_id"] == "itm_existing"
    assert messages[2]["item_id"] == "itm_appended"
    _assert_unique_non_empty_ids(messages)


def test_extended_tool_calls_receive_isolated_ids():
    messages = ChatMessages()
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "calling tools",
                "item_id": "itm_assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "first", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "second", "arguments": "{}"},
                    },
                ],
            }
        ]
    )

    assert [item["type"] for item in messages] == [
        "message",
        "function_call",
        "function_call",
    ]
    assert messages[0]["item_id"] == "itm_assistant"
    _assert_unique_non_empty_ids(messages)


def test_setitem_can_reuse_the_replaced_id_but_not_other_ids():
    messages = ChatMessages(
        [
            _message("first", item_id="itm_first"),
            _message("second", item_id="itm_second"),
        ]
    )

    messages[0] = _message("replacement", item_id="itm_first")
    assert messages[0]["item_id"] == "itm_first"

    messages[0] = _message("collision", item_id="itm_second")
    assert messages[0]["item_id"] != "itm_second"
    _assert_unique_non_empty_ids(messages)


def test_extend_keeps_partial_mutation_when_an_item_fails():
    messages = ChatMessages([_message("existing", item_id="itm_existing")])

    with pytest.raises(TypeError):
        messages.extend(
            [
                _message("kept", item_id="itm_kept"),
                "not-a-mapping",
                _message("never-added"),
            ]
        )

    assert [item["content"] for item in messages] == ["existing", "kept"]
    _assert_unique_non_empty_ids(messages)


def test_extend_keeps_partial_mutation_when_a_generator_fails():
    messages = ChatMessages()

    def broken_items():
        yield _message("generated", item_id="itm_generated")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        messages.extend(broken_items())

    assert [item["content"] for item in messages] == ["generated"]
    _assert_unique_non_empty_ids(messages)


def test_extend_rescans_when_a_generator_appends_between_yields():
    messages = ChatMessages()

    def items():
        yield _message("first")
        messages.append(_message("injected", item_id="itm_mid"))
        yield _message("yielded", item_id="itm_mid")

    messages.extend(items())

    assert [item["content"] for item in messages] == ["first", "injected", "yielded"]
    assert messages[1]["item_id"] == "itm_mid"
    assert messages[2]["item_id"] != "itm_mid"
    _assert_unique_non_empty_ids(messages)


def test_extend_rescans_when_a_generator_mutates_an_existing_id():
    messages = ChatMessages()

    def items():
        yield _message("first", item_id="itm_first")
        messages[0]["item_id"] = "itm_changed"
        yield _message("second", item_id="itm_changed")

    messages.extend(items())

    assert [item["content"] for item in messages] == ["first", "second"]
    assert messages[0]["item_id"] == "itm_changed"
    assert messages[1]["item_id"] != "itm_changed"
    _assert_unique_non_empty_ids(messages)


class _CountingItems(list):
    def __init__(self, items=()):
        super().__init__(items)
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        return super().__iter__()


def test_extend_scans_existing_items_once_per_stable_sequence_batch():
    messages = ChatMessages()
    messages._items = _CountingItems(
        _message(f"existing-{index}", item_id=f"itm_existing_{index}")
        for index in range(50)
    )

    messages.extend([_message(f"new-{index}") for index in range(25)])

    assert messages._items.iteration_count == 1
    assert len(messages) == 75
    _assert_unique_non_empty_ids(messages)


def test_constructor_copy_and_prefix_copy_scan_ids_once(monkeypatch):
    scans: list[int] = []
    original_item_id_set = ChatMessages._item_id_set

    def counting_item_id_set(self, *args, **kwargs):
        scans.append(len(self._items))
        return original_item_id_set(self, *args, **kwargs)

    monkeypatch.setattr(ChatMessages, "_item_id_set", counting_item_id_set)

    source = ChatMessages(
        [
            _message(f"existing-{index}", item_id=f"itm_existing_{index}")
            for index in range(50)
        ]
    )
    assert scans == [0]

    scans.clear()
    copied = source.copy()
    assert scans == [0]
    assert len(copied) == 50
    _assert_unique_non_empty_ids(copied)

    scans.clear()
    through = source.through_item(source[10]["item_id"])
    assert scans == [0]
    assert len(through) == 11
    assert _item_ids(through) == _item_ids(source)[:11]
    _assert_unique_non_empty_ids(through)


@pytest.mark.parametrize("base", [list, tuple])
def test_extend_rescans_effectful_sequence_subclasses(base):
    messages = ChatMessages()

    class EffectfulItems(base):
        def __iter__(self):
            yield _message("first", item_id="itm_a")
            messages[0]["item_id"] = "itm_changed"
            yield _message("second", item_id="itm_changed")

    messages.extend(EffectfulItems())

    assert messages[0]["item_id"] == "itm_changed"
    assert messages[1]["item_id"] != "itm_changed"
    _assert_unique_non_empty_ids(messages)
