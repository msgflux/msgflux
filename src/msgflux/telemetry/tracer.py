import threading

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import NoOpTracerProvider

from msgflux.envs import envs
from msgflux.logger import logger


class TracerManager:
    def __init__(self):
        self._configured = False
        self._lock = threading.RLock()
        self._tracer = None

    def _build_sampler(self) -> None:
        """Optional sampler based on environment ratio."""
        ratio = getattr(envs, "telemetry_sampling_ratio", None)
        if ratio is not None:
            try:
                sampler = ParentBased(TraceIdRatioBased(float(ratio)))
                return sampler
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid sampling ratio `{ratio}`, "
                    "falling back to default sampler."
                )
        return None

    def configure_tracer(self):
        """Configure OpenTelemetry tracer. Thread-safe and idempotent."""
        with self._lock:
            if self._configured:
                return

            if not envs.telemetry_requires_trace:
                logger.debug("Tracing disabled, configuring NoOp tracer")
                no_op_provider = NoOpTracerProvider()
                trace.set_tracer_provider(no_op_provider)
                self._tracer = trace.get_tracer("msgflux.telemetry")
                self._configured = True
                return

            # Resource and provider setup
            attributes = {SERVICE_NAME: "msgflux-telemetry"}
            resource = Resource.create(attributes)
            sampler = self._build_sampler()
            if sampler:
                provider = TracerProvider(resource=resource, sampler=sampler)
            else:
                provider = TracerProvider(resource=resource)

            exporter_type = envs.telemetry_span_exporter_type.lower()
            if exporter_type == "otlp":
                otlp_exporter = OTLPSpanExporter(endpoint=envs.telemetry_otlp_endpoint)
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                logger.debug(
                    "Configured OTLP exporter with endpoint: "
                    f"{envs.telemetry_otlp_endpoint}"
                )
            elif exporter_type == "console":
                console_exporter = ConsoleSpanExporter()
                processor = BatchSpanProcessor(console_exporter)
                provider.add_span_processor(processor)
                logger.debug("Configured Console exporter")
            else:
                # Unknown exporter: warn and fallback to NoOp
                logger.warning(
                    f"Unknown exporter type `{envs.telemetry_span_exporter_type}` "
                    "defaulting to NoOp tracer"
                )
                no_op_provider = NoOpTracerProvider()
                trace.set_tracer_provider(no_op_provider)
                self._tracer = trace.get_tracer("msgflux.telemetry")
                self._configured = True
                return

            # Finalize provider and cache tracer
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("msgflux.telemetry")
            self._configured = True

    def get_tracer(self):
        """Get the configured tracer, initializing on first access."""
        with self._lock:
            if not self._configured:
                self.configure_tracer()
            return self._tracer


# Singleton instance
tracer_manager = TracerManager()

def get_tracer():
    """Convenience function to retrieve the global tracer."""
    return tracer_manager.get_tracer()
