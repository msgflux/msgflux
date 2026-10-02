"""Explicit progress messages for models without a trusted commentary phase."""

from msgflux.runtime.events import EventType, _is_capturing_events, emit_event


class SendUserMessageTool:
    """Send a brief user-visible update while continuing the task.

    Args:
        message: Progress text for the user. Do not include private reasoning.
    """

    name = "send_user_message"
    display_name = "Send User Message"
    annotations = {"message": str, "return": str}

    def __call__(self, message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must contain non-empty user-visible text")
        if not _is_capturing_events():
            raise RuntimeError("send_user_message requires an active event stream")
        emit_event(EventType.COMMENTARY_DELTA, {"delta": message})
        return "Message sent to user. Continue the task."


__all__ = ["SendUserMessageTool"]
