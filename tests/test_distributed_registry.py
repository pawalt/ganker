from ganker.distributed.registry import (
    EndpointAddress,
    InMemoryRunRegistry,
    RoleEndpoint,
    RoleKey,
)


def test_role_endpoint_round_trips_as_json_like_dict():
    endpoint = RoleEndpoint(
        deployment_id="dev",
        run_id="run-1",
        role="trainer",
        rank=0,
        protocol="tcp",
        addresses=(EndpointAddress(family="ipv6", host="fdaa::1", port=26600),),
        status="ready",
        epoch=3,
        region="us-east-1",
        metadata={"kind": "fake"},
    )

    decoded = RoleEndpoint.from_dict(endpoint.as_dict())

    assert decoded == endpoint
    assert decoded.storage_key == "ganker:dev:run-1:trainer:0"
    assert decoded.target() == "tcp://[fdaa::1]:26600"


def test_registry_lists_role_targets_in_rank_order():
    registry = InMemoryRunRegistry()
    for rank in (1, 0):
        registry.put(
            RoleEndpoint(
                deployment_id="dev",
                run_id="run-1",
                role="trainer",
                rank=rank,
                protocol="tcp",
                addresses=(
                    EndpointAddress(
                        family="ipv6",
                        host=f"fdaa::{rank + 1}",
                        port=26600,
                    ),
                ),
                status="ready",
                region="us-east-1",
            )
        )

    assert registry.get(RoleKey("dev", "run-1", "trainer", 0)).rank == 0
    assert registry.targets(deployment_id="dev", run_id="run-1", role="trainer") == [
        "tcp://[fdaa::1]:26600",
        "tcp://[fdaa::2]:26600",
    ]
