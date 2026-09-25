"""MCP client exceptions."""


class MCPError(Exception):
    """Base exception for MCP client errors."""

    pass


class MCPTimeoutError(MCPError):
    """Raised when MCP operations timeout."""

    pass


class MCPToolError(MCPError):
    """Raised when tool execution fails."""

    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    pass


class MCPInputRequiredError(MCPError):
    """A server needs client input before it can complete a request."""

    def __init__(self, input_requests, request_state=None):
        super().__init__("MCP request requires additional client input")
        self.input_requests = input_requests
        self.request_state = request_state
