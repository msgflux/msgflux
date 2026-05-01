class ChannelError(Exception):
    status_code = 400
    code = "channel_error"
    error_type = "invalid_request"

    def __init__(self, message: str, *, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        self.headers = headers or {}


class AgentNotFoundError(ChannelError):
    status_code = 404
    code = "agent_not_found"


class UnauthorizedError(ChannelError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(ChannelError):
    status_code = 403
    code = "forbidden"


class PayloadTooLargeError(ChannelError):
    status_code = 413
    code = "payload_too_large"


class RequestTimeoutError(ChannelError):
    status_code = 504
    code = "request_timeout"


class RateLimitExceededError(ChannelError):
    status_code = 429
    code = "rate_limit_exceeded"

    def __init__(self, message: str, *, retry_after_s: float | None = None):
        headers = {}
        if retry_after_s is not None:
            headers["Retry-After"] = str(max(1, int(retry_after_s)))
        super().__init__(message, headers=headers)
