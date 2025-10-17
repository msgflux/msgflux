"""Authentication providers for MCP protocol."""

from msgflux.protocols.mcp.auth.base import BaseAuth
from msgflux.protocols.mcp.auth.bearer import BearerTokenAuth
from msgflux.protocols.mcp.auth.apikey import APIKeyAuth
from msgflux.protocols.mcp.auth.basic import BasicAuth
from msgflux.protocols.mcp.auth.oauth2 import OAuth2Auth
from msgflux.protocols.mcp.auth.custom import CustomHeaderAuth

__all__ = [
    "BaseAuth",
    "BearerTokenAuth",
    "APIKeyAuth",
    "BasicAuth",
    "OAuth2Auth",
    "CustomHeaderAuth",
]
