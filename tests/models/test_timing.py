from msgflux.models.timing import ModelRequestTimer


def test_request_timer_records_first_output_and_latency_once():
    ticks = iter([1_000_000, 3_000_000, 4_000_000])
    timer = ModelRequestTimer(clock_ns=lambda: next(ticks))

    timer.mark_first_output()
    timer.mark_first_output()

    assert timer.finish() == {
        "source": "provider",
        "latency_ms": 3.0,
        "ttft_ms": 2.0,
    }
    assert timer.finish() == {
        "source": "provider",
        "latency_ms": 3.0,
        "ttft_ms": 2.0,
    }


def test_request_timer_omits_ttft_without_observable_output():
    ticks = iter([1_000_000, 2_500_000])
    timer = ModelRequestTimer(source="cache", clock_ns=lambda: next(ticks))

    assert timer.finish() == {
        "source": "cache",
        "latency_ms": 1.5,
    }
