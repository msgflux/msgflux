"""Small helpers for consistent public exception messages."""


def exception_message(error: BaseException) -> str:
    """Preserve the reason, using the exception class when its message is blank."""
    message = str(error)
    return message if message.strip() else type(error).__name__
