from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.tools.config import tool_config
from msgflux.utils.chat import ChatBlock

IMAGE = ChatBlock.image("data:image/png;base64,AA==", detail="low")


@pytest.fixture(params=["memory", "sqlite"])
def inbox(request, tmp_path):
    store = (
        InMemoryAgentInboxStore()
        if request.param == "memory"
        else SQLiteAgentInboxStore(path=str(tmp_path / "inbox.db"))
    )
    yield AgentInbox(store=store, namespace="a", thread_id="t", run_id="r")
    if request.param == "sqlite":
        store.close()


def test_multimodal_roundtrip_release_and_provenance(inbox):
    original = [ChatBlock.text("look <here>"), deepcopy(IMAGE)]
    user = inbox.user_message(original)
    tool = inbox.message(
        [IMAGE], description="Image from tool", source="read_image", ref='call"1'
    )
    original[1]["image_url"]["url"] = "changed"
    reader = inbox.fork()
    claimed = reader.claim()
    reader.release()
    assert reader.claim() == claimed
    messages = reader.render_messages(claimed)
    assert [m["role"] for m in messages] == ["user", "user"]
    assert messages[0]["content"][0]["text"] == "<incoming_user_message>"
    assert messages[0]["content"][1]["text"] == "look &lt;here&gt;"
    assert messages[0]["content"][2] == IMAGE
    assert messages[1]["content"][1] == IMAGE
    assert messages[1]["metadata"] == {
        "inbox_origin": "read_image",
        "inbox_ref": 'call"1',
    }
    assert "incoming_user_message" not in messages[1]["content"][0]["text"]
    reader.ack([user.notification_id, tool.notification_id])
    assert inbox.peek() == []


def test_clear_only_user_content(inbox):
    inbox.user_message([IMAGE])
    inbox.message("tool text", description="attachment", source="tool")
    inbox.publish({"source": "task", "status": "done"})
    assert inbox.clear_user_messages() == 1
    assert [n.source for n in inbox.peek()] == ["incoming_message", "task"]


@pytest.mark.parametrize(
    "content",
    [
        [],
        42,
        [{"type": "input_image", "image_url": "https://example.com/i"}],
        [ChatBlock.image("/etc/image.png")],
        [ChatBlock.image("file:///image.png")],
        [{"type": "text", "text": 1}],
        [ChatBlock.image("https://example.com/i", detail="bad")],
    ],
)
def test_invalid_content_not_published(content):
    inbox = AgentInbox(store=InMemoryAgentInboxStore())
    with pytest.raises((TypeError, ValueError)):
        inbox.user_message(content)
    assert inbox.peek() == []


def test_mapping_publication_validates_before_store(inbox):
    with pytest.raises(ValueError, match="description"):
        inbox.publish(
            {
                "source": "incoming_message",
                "metadata": {"origin": "tool", "content": [IMAGE]},
            }
        )
    assert inbox.peek() == []


def test_verbose_multimodal_drain(capsys):
    inbox = AgentInbox(store=InMemoryAgentInboxStore(), verbose=True)
    inbox.user_message([IMAGE])
    inbox.drain()
    assert "notification_drain" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [True, False])
async def test_tool_image_follows_result_and_survives_checkpoint(inbox, asynchronous):
    published = []

    @tool_config(runtime_inputs=["handle"], retry=False)
    def show_image(*, handle) -> str:
        """Publish an image to the conversation."""
        published.append(
            handle.get_notification().message(
                [IMAGE], description="Image from this tool call"
            )
        )
        return "Image follows as a user-role message."

    model = Mock(model_type="chat_completion")
    checkpoint = InMemoryCheckpointStore()
    agent = Agent(
        name="a",
        model=model,
        tools=[show_image],
        agent_inbox=inbox,
        checkpoint_store=checkpoint,
    )
    calls = ToolCallAggregator()
    calls.process(0, "image:1", "show_image", "{}")
    first, final = ModelResponse(), ModelResponse()
    first.set_response_type("tool_call")
    first.add(calls)
    final.set_response_type("text_generation")
    final.add("done")
    agent.generator.aforward = AsyncMock(side_effect=[first, final])
    agent.generator.forward = Mock(side_effect=[first, final])
    scope = ExecutionScope(namespace="a", thread_id="t", run_id="r")
    messages = ChatMessages()
    if asynchronous:
        assert await agent.acall("show", messages=messages, scope=scope) == "done"
    else:
        assert agent("show", messages=messages, scope=scope) == "done"
    state = checkpoint.load_state("a", "t", "r")
    history = state["messages"]["items"]
    result_index = next(
        i for i, m in enumerate(history) if m.get("type") == "function_call_output"
    )
    image_index = next(
        i
        for i, m in enumerate(history)
        if isinstance(m.get("content"), list) and IMAGE in m["content"]
    )
    assert image_index > result_index
    assert history[image_index]["metadata"]["inbox_ref"] == "image:1"
    assert inbox.peek() == []
    assert "data:image/png;base64,AA==" in str(state)
    # Simulate the notification surviving an interrupted acknowledgment.
    inbox.publish(published[0])
    recovered = ChatMessages(state["messages"]["items"])
    recovered.update_metadata(state["messages"]["metadata"])
    assert (
        agent._prepare_inbox_delivery(inbox, recovered, inbox.claim(), drain=True) == []
    )
    inbox.ack(inbox.delivered_ids())
    restored = recovered.to_items()
    assert (
        sum(
            isinstance(item.get("content"), list) and IMAGE in item["content"]
            for item in restored
        )
        == 1
    )
    assert inbox.peek() == []
