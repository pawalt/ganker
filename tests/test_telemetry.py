import pytest

from ganker.components import TelemetryLedger
from ganker.contracts import Usage, UsageEvent
from ganker.errors import InvalidRequestError


def test_telemetry_ledger_records_and_summarizes_by_source():
    ledger = TelemetryLedger()

    ledger.record(
        UsageEvent(
            request_id="req-1",
            run_id="run-1",
            event_source="trainer",
            usage=Usage(input_tokens=4),
        )
    )
    ledger.record(
        UsageEvent(
            request_id="req-2",
            run_id="run-1",
            event_source="rollout",
            usage=Usage(input_tokens=2, output_tokens=3, samples=1),
        )
    )

    summary = ledger.summary("run-1")

    assert summary.event_count == 2
    assert summary.total.input_tokens == 6
    assert summary.total.output_tokens == 3
    assert summary.total.samples == 1
    assert [source.event_source for source in summary.by_source] == ["rollout", "trainer"]


def test_telemetry_ledger_requires_core_event_fields():
    ledger = TelemetryLedger()

    with pytest.raises(InvalidRequestError):
        ledger.record(UsageEvent(request_id="req-1", run_id="", event_source="trainer", usage=Usage()))
