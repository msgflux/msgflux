from typing import Any, Dict, List, Optional, Union

from msgspec import Struct


class ToolDefinitions(Struct, kw_only=True):
    """Container for runtime tool metadata shared across agent/provider flows."""

    schemas: Optional[List[Dict[str, Any]]] = None
    annotations: Optional[Dict[str, Dict[str, Any]]] = None
    choice: Optional[Union[str, Dict[str, Any]]] = None
