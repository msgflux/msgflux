"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.web_fetch import WebFetch
from msgflux.tools.builtin.web_search import WebSearch
from msgflux.tools.builtin.weather import Weather

__all__ = ["WebFetch", "WebSearch", "Weather"]
