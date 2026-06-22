from typing import Any

from ganker.distributed import monarch
from ganker.distributed.registry import EndpointAddress, RoleEndpoint


def _endpoint() -> RoleEndpoint:
    return RoleEndpoint(
        deployment_id="dev",
        run_id="run-1",
        role="trainer",
        rank=0,
        protocol="tcp",
        addresses=(EndpointAddress(family="ipv6", host="fdaa::1", port=26600),),
        status="ready",
        region="us-east-1",
    )


def test_attach_role_endpoints_separates_worker_protocol_from_controller_transport(
    monkeypatch,
):
    calls: list[tuple[str, Any]] = []

    def fake_enable_transport(transport: str) -> None:
        calls.append(("enable_transport", transport))

    def fake_attach_to_workers(**kwargs: Any) -> str:
        calls.append(("attach_to_workers", kwargs))
        return "mesh"

    monkeypatch.setattr(monarch, "enable_transport", fake_enable_transport)
    monkeypatch.setattr(monarch, "attach_to_workers", fake_attach_to_workers)

    result = monarch.attach_role_endpoints(
        [_endpoint()],
        name="trainer",
        family="ipv6",
        protocol="tcp",
        transport="tcp://[fdaa::2]:26610",
    )

    assert result == "mesh"
    assert calls == [
        ("enable_transport", "tcp://[fdaa::2]:26610"),
        (
            "attach_to_workers",
            {
                "ca": "trust_all_connections",
                "workers": ["tcp://[fdaa::1]:26600"],
                "name": "trainer",
            },
        ),
    ]


def test_attach_role_endpoints_can_reuse_existing_transport(monkeypatch):
    calls: list[tuple[str, Any]] = []

    def fake_enable_transport(transport: str) -> None:
        calls.append(("enable_transport", transport))

    def fake_attach_to_workers(**kwargs: Any) -> str:
        calls.append(("attach_to_workers", kwargs))
        return "mesh"

    monkeypatch.setattr(monarch, "enable_transport", fake_enable_transport)
    monkeypatch.setattr(monarch, "attach_to_workers", fake_attach_to_workers)

    result = monarch.attach_role_endpoints(
        [_endpoint()],
        name="trainer",
        family="ipv6",
        transport=None,
    )

    assert result == "mesh"
    assert calls == [
        (
            "attach_to_workers",
            {
                "ca": "trust_all_connections",
                "workers": ["tcp://[fdaa::1]:26600"],
                "name": "trainer",
            },
        ),
    ]
