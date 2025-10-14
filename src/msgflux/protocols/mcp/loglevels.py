"""MCP logging levels."""

from enum import Enum


class LogLevel(str, Enum):
    """MCP logging levels."""

    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"
