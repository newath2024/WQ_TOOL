from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from storage.service_runtime_store import ServiceRuntimeStore


@dataclass(slots=True, frozen=True)
class RuntimeLock:
    store: ServiceRuntimeStore
    service_name: str
    service_run_id: str
    lease_seconds: int
    owner_token: str
    pid: int
    hostname: str

    @classmethod
    def create(cls, store: ServiceRuntimeStore, *, service_name: str, service_run_id: str, lease_seconds: int) -> "RuntimeLock":
        return cls(
            store=store,
            service_name=service_name,
            service_run_id=service_run_id,
            lease_seconds=lease_seconds,
            owner_token=uuid4().hex,
            pid=os.getpid(),
            hostname=socket.gethostname(),
        )

    def acquire(self, *, status: str) -> bool:
        now = _utcnow()
        return self.store.try_acquire_lease(
            service_name=self.service_name,
            owner_token=self.owner_token,
            service_run_id=self.service_run_id,
            pid=self.pid,
            hostname=self.hostname,
            status=status,
            now=now,
            lock_expires_at=_shift(now, self.lease_seconds),
        )

    def renew(self) -> bool:
        now = _utcnow()
        return self.store.renew_lease(
            service_name=self.service_name,
            owner_token=self.owner_token,
            lock_expires_at=_shift(now, self.lease_seconds),
            updated_at=now,
        )

    def release(self, *, status: str) -> None:
        self.store.release_lease(
            service_name=self.service_name,
            owner_token=self.owner_token,
            status=status,
            updated_at=_utcnow(),
        )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _shift(timestamp: str, seconds: int) -> str:
    return (datetime.fromisoformat(timestamp) + timedelta(seconds=seconds)).isoformat()
