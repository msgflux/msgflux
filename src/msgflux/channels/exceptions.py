class ChannelError(Exception):
    status_code = 400
    code = "channel_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AgentNotFoundError(ChannelError):
    status_code = 404
    code = "agent_not_found"
