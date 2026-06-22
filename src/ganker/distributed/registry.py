"""Endpoint registry contracts for distributed Monarch roles."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ganker.errors import InvalidRequestError, NotFoundError


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RoleKey:
    deployment_id: str
    run_id: str
    role: str
    rank: int = 0

    def __post_init__(self) -> None:
        if not self.deployment_id:
            raise InvalidRequestError("deployment_id is required")
        if not self.run_id:
            raise InvalidRequestError("run_id is required")
        if not self.role:
            raise InvalidRequestError("role is required")
        if self.rank < 0:
            raise InvalidRequestError("rank must be non-negative")

    @property
    def storage_key(self) -> str:
        return f"ganker:{self.deployment_id}:{self.run_id}:{self.role}:{self.rank}"

    @classmethod
    def from_storage_key(cls, value: str) -> "RoleKey":
        parts = value.split(":")
        if len(parts) != 5 or parts[0] != "ganker":
            raise InvalidRequestError(f"invalid role key: {value}")
        return cls(
            deployment_id=parts[1],
            run_id=parts[2],
            role=parts[3],
            rank=int(parts[4]),
        )


@dataclass(frozen=True)
class EndpointAddress:
    family: str
    host: str
    port: int

    def __post_init__(self) -> None:
        if self.family not in {"ipv4", "ipv6", "ipc", "inproc"}:
            raise InvalidRequestError(f"unsupported address family: {self.family}")
        if not self.host:
            raise InvalidRequestError("address host is required")
        if self.port < 0:
            raise InvalidRequestError("address port must be non-negative")

    @property
    def host_for_uri(self) -> str:
        if self.family == "ipv6" and not self.host.startswith("["):
            return f"[{self.host}]"
        return self.host

    def target(self, protocol: str) -> str:
        if self.family in {"ipc", "inproc"}:
            return f"{self.family}://{self.host}"
        return f"{protocol}://{self.host_for_uri}:{self.port}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "host": self.host,
            "port": self.port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EndpointAddress":
        return cls(
            family=str(data["family"]),
            host=str(data["host"]),
            port=int(data.get("port", 0)),
        )


@dataclass(frozen=True)
class RoleEndpoint:
    deployment_id: str
    run_id: str
    role: str
    rank: int
    protocol: str
    addresses: tuple[EndpointAddress, ...]
    status: str = "starting"
    epoch: int = 0
    region: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    last_heartbeat: str = field(default_factory=utc_now_iso)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        RoleKey(self.deployment_id, self.run_id, self.role, self.rank)
        if not self.protocol:
            raise InvalidRequestError("endpoint protocol is required")
        if not self.addresses:
            raise InvalidRequestError("endpoint requires at least one address")
        if self.epoch < 0:
            raise InvalidRequestError("endpoint epoch must be non-negative")

    @property
    def key(self) -> RoleKey:
        return RoleKey(self.deployment_id, self.run_id, self.role, self.rank)

    @property
    def storage_key(self) -> str:
        return self.key.storage_key

    def preferred_address(self, *, family: str | None = None) -> EndpointAddress:
        if family is not None:
            for address in self.addresses:
                if address.family == family:
                    return address
            raise NotFoundError(
                f"{self.role}:{self.rank} has no {family} address in {self.region}"
            )
        return self.addresses[0]

    def target(self, *, family: str | None = None) -> str:
        return self.preferred_address(family=family).target(self.protocol)

    def heartbeat(self, *, status: str | None = None) -> "RoleEndpoint":
        return RoleEndpoint(
            deployment_id=self.deployment_id,
            run_id=self.run_id,
            role=self.role,
            rank=self.rank,
            protocol=self.protocol,
            addresses=self.addresses,
            status=status or self.status,
            epoch=self.epoch,
            region=self.region,
            started_at=self.started_at,
            last_heartbeat=utc_now_iso(),
            metadata=dict(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "run_id": self.run_id,
            "role": self.role,
            "rank": self.rank,
            "protocol": self.protocol,
            "addresses": [address.as_dict() for address in self.addresses],
            "status": self.status,
            "epoch": self.epoch,
            "region": self.region,
            "started_at": self.started_at,
            "last_heartbeat": self.last_heartbeat,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoleEndpoint":
        return cls(
            deployment_id=str(data["deployment_id"]),
            run_id=str(data["run_id"]),
            role=str(data["role"]),
            rank=int(data["rank"]),
            protocol=str(data["protocol"]),
            addresses=tuple(
                EndpointAddress.from_dict(address)
                for address in data.get("addresses", [])
            ),
            status=str(data.get("status", "starting")),
            epoch=int(data.get("epoch", 0)),
            region=str(data.get("region", "")),
            started_at=str(data.get("started_at") or utc_now_iso()),
            last_heartbeat=str(data.get("last_heartbeat") or utc_now_iso()),
            metadata={str(k): str(v) for k, v in data.get("metadata", {}).items()},
        )


class _ObjectStore(Protocol):
    def put(self, key: str, value: dict[str, Any]) -> Any:
        ...

    def get(self, key: str, default: Any | None = None) -> Any:
        ...

    def keys(self) -> Iterable[str]:
        ...


class MappingRunRegistry:
    """Registry adapter for dict-like stores such as Modal Dict.

    Values are stored as plain dictionaries so the backing store does not need
    to know about Ganker classes.
    """

    def __init__(self, store: MutableMapping[str, dict[str, Any]] | _ObjectStore):
        self._store = store

    def put(self, endpoint: RoleEndpoint) -> RoleEndpoint:
        payload = endpoint.as_dict()
        put = getattr(self._store, "put", None)
        if put is not None:
            put(endpoint.storage_key, payload)
        else:
            self._store[endpoint.storage_key] = payload  # type: ignore[index]
        return endpoint

    def heartbeat(self, key: RoleKey, *, status: str | None = None) -> RoleEndpoint:
        endpoint = self.get(key).heartbeat(status=status)
        return self.put(endpoint)

    def get(self, key: RoleKey) -> RoleEndpoint:
        storage_key = key.storage_key
        getter = getattr(self._store, "get", None)
        if getter is not None:
            payload = getter(storage_key)
        else:
            payload = self._store.get(storage_key)  # type: ignore[union-attr]
        if payload is None:
            raise NotFoundError(f"role endpoint not found: {storage_key}")
        return RoleEndpoint.from_dict(payload)

    def list(self, *, deployment_id: str, run_id: str) -> list[RoleEndpoint]:
        prefix = f"ganker:{deployment_id}:{run_id}:"
        keys = list(self._store.keys())
        endpoints: list[RoleEndpoint] = []
        for key in sorted(str(item) for item in keys if str(item).startswith(prefix)):
            endpoints.append(RoleEndpoint.from_dict(self._store.get(key)))  # type: ignore[arg-type, union-attr]
        return endpoints

    def targets(
        self,
        *,
        deployment_id: str,
        run_id: str,
        role: str,
        protocol: str = "tcp",
        family: str | None = None,
    ) -> list[str]:
        endpoints = [
            endpoint
            for endpoint in self.list(deployment_id=deployment_id, run_id=run_id)
            if endpoint.role == role and endpoint.protocol == protocol
        ]
        return [endpoint.target(family=family) for endpoint in sorted(endpoints, key=lambda e: e.rank)]


class InMemoryRunRegistry(MappingRunRegistry):
    def __init__(self):
        super().__init__({})
