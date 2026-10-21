"""HTTP instrumentation using the host's existing OpenTelemetry providers."""

from litestar.plugins.opentelemetry import OpenTelemetryConfig, OpenTelemetryPlugin
from opentelemetry import metrics, trace

_IDENTITY_FIELDS = ("agent_id", "thread_id", "run_id", "request_id")


def _span_details(scope):
    # Routing has not run yet. Never use concrete thread IDs in span names.
    return scope.get("method", "HTTP"), {}


def service_telemetry_plugin(instance_id: str) -> OpenTelemetryPlugin:
    def request_hook(span, _scope):
        if span.is_recording():
            span.set_attribute("msgflux.service.instance_id", instance_id)

    return OpenTelemetryPlugin(
        OpenTelemetryConfig(
            tracer_provider=trace.get_tracer_provider(),
            meter_provider=metrics.get_meter_provider(),
            scope_span_details_extractor=_span_details,
            server_request_hook_handler=request_hook,
            exclude_spans=["receive", "send"],
            scopes=("http",),
            http_capture_headers_server_request=[],
            http_capture_headers_server_response=[],
            http_capture_headers_sanitize_fields=[
                "authorization",
                "proxy-authorization",
                "cookie",
                "set-cookie",
                ".*api.*key.*",
                ".*token.*",
            ],
        )
    )


async def annotate_http_response(message, scope) -> None:
    """Use the resolved route and IDs after routing, including on auth errors."""
    if message["type"] != "http.response.start":
        return
    span = trace.get_current_span()
    if not span.is_recording():
        return
    if template := scope.get("path_template"):
        span.update_name(f"{scope['method']} {template}")
        span.set_attribute("http.route", template)
    _annotate_identities(span, scope.get("path_params", {}))


def annotate_http_result(value) -> None:
    """Attach receipt identities without recording response content."""
    span = trace.get_current_span()
    if span.is_recording():
        _annotate_identities(
            span, {name: getattr(value, name, None) for name in _IDENTITY_FIELDS}
        )


def _annotate_identities(span, identities):
    for name in _IDENTITY_FIELDS:
        identity = identities.get(name)
        if isinstance(identity, str) and identity:
            span.set_attribute(f"msgflux.{name}", identity)
