import threading

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import NoOpTracerProvider

from msgflux.envs import envs
from msgflux.logger import logger


class TracerManager:
    def __init__(self):
        self._configured = False
        self._lock = threading.Lock()
        self._tracer = None

    def configure_tracer(self):
        """Configure OpenTelemetry tracer."""
        with self._lock: # Acquire the lock
            if self._configured:
                return

            if not envs.telemetry_requires_trace:
                logger.debug("Tracing disabled, configuring NoOp tracer")
                no_op_provider = NoOpTracerProvider()
                trace.set_tracer_provider(no_op_provider)
                self._configured = True
                return

            attributes = {SERVICE_NAME: "msgflux-telemetry"}
            resource = Resource.create(attributes)
            provider = TracerProvider(resource=resource)

            if envs.telemetry_span_exporter_type.lower() == "otlp":
                otlp_exporter = OTLPSpanExporter(endpoint=envs.telemetry_otlp_endpoint)
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                logger.debug(
                    "Configured OTLP exporter with endpoint: "
                    f"{envs.telemetry_otlp_endpoint}"
                )
            elif envs.telemetry_span_exporter_type.lower() == "console":
                console_exporter = ConsoleSpanExporter()
                processor = BatchSpanProcessor(console_exporter)
                provider.add_span_processor(processor)
                logger.debug("Configured Console exporter")

            trace.set_tracer_provider(provider)
            self._configured = True

    def get_tracer(self):
        """Get the configured tracer."""
        with self._lock:  # Acquire the lock
            if not self._configured:
                self.configure_tracer()
        return trace.get_tracer("msgflux.telemetry")

tracer_manager = TracerManager()

def get_tracer():
    return tracer_manager.get_tracer()
