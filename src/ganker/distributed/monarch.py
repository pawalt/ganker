"""Monarch attach helpers for distributed role workers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from monarch.actor import Future, attach_to_workers, enable_transport

from ganker.distributed.registry import MappingRunRegistry, RoleEndpoint
from ganker.errors import InvalidRequestError


def endpoint_targets(
    endpoints: Sequence[RoleEndpoint],
    *,
    protocol: str = "tcp",
    family: str | None = None,
) -> list[str]:
    selected = [endpoint for endpoint in endpoints if endpoint.protocol == protocol]
    if not selected:
        raise InvalidRequestError(f"no {protocol} endpoints were provided")
    return [
        endpoint.target(family=family)
        for endpoint in sorted(selected, key=lambda item: item.rank)
    ]


def attach_role_endpoints(
    endpoints: Sequence[RoleEndpoint],
    *,
    name: str,
    family: str | None = None,
    protocol: str = "tcp",
    transport: str | None = "tcp",
) -> Any:
    """Attach to Monarch worker listeners described by endpoint metadata."""

    if transport is not None:
        enable_transport(transport)
    workers = cast(list[str | Future[str]], endpoint_targets(endpoints, protocol=protocol, family=family))
    return attach_to_workers(
        ca="trust_all_connections",
        workers=workers,
        name=name,
    )


def attach_role_from_registry(
    registry: MappingRunRegistry,
    *,
    deployment_id: str,
    run_id: str,
    role: str,
    name: str | None = None,
    family: str | None = None,
    protocol: str = "tcp",
    transport: str | None = "tcp",
) -> Any:
    """Attach a HostMesh to all endpoints for one role in the registry."""

    endpoints = [
        endpoint
        for endpoint in registry.list(deployment_id=deployment_id, run_id=run_id)
        if endpoint.role == role
    ]
    if not endpoints:
        raise InvalidRequestError(f"no endpoints registered for role {role!r}")
    return attach_role_endpoints(
        endpoints,
        name=name or role,
        family=family,
        protocol=protocol,
        transport=transport,
    )
