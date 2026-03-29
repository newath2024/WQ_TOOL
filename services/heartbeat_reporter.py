from __future__ import annotations

import json
from datetime import UTC, datetime

from services.models import ServiceTickOutcome
from storage.models import ServiceRuntimeRecord
from storage.service_runtime_store import ServiceRuntimeStore


class HeartbeatReporter:
    def __init__(self, store: ServiceRuntimeStore) -> None:
        self.store = store

    def record_tick(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        outcome: ServiceTickOutcome,
        tick_id: int,
    ) -> ServiceRuntimeRecord:
        now = datetime.now(UTC).isoformat()
        counters = json.loads(runtime.counters_json or "{}")
        counters["generated"] = int(counters.get("generated", 0)) + outcome.generated_count
        counters["submitted"] = int(counters.get("submitted", 0)) + outcome.submitted_count
        counters["completed"] = int(counters.get("completed", 0)) + outcome.completed_count
        counters["failed"] = int(counters.get("failed", 0)) + outcome.failed_count
        counters["quarantined"] = int(counters.get("quarantined", 0)) + outcome.quarantined_count
        self.store.update_state(
            runtime.service_name,
            status=outcome.status,
            tick_id=tick_id,
            active_batch_id=outcome.active_batch_id,
            pending_job_count=outcome.pending_job_count,
            consecutive_failures=0,
            cooldown_until=outcome.cooldown_until,
            last_heartbeat_at=now,
            last_success_at=now if outcome.last_error is None else runtime.last_success_at,
            last_error=outcome.last_error,
            persona_url=outcome.persona_url,
            counters_json=json.dumps(counters, sort_keys=True),
            updated_at=now,
        )
        refreshed = self.store.get_state(runtime.service_name)
        if refreshed is None:
            raise RuntimeError(f"Missing service runtime row after heartbeat for {runtime.service_name}")
        return refreshed

    def record_failure(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        last_error: str,
        consecutive_failures: int,
        cooldown_until: str | None,
    ) -> ServiceRuntimeRecord:
        now = datetime.now(UTC).isoformat()
        self.store.update_state(
            runtime.service_name,
            status="cooldown" if cooldown_until else "error",
            tick_id=tick_id,
            consecutive_failures=consecutive_failures,
            cooldown_until=cooldown_until,
            last_heartbeat_at=now,
            last_error=last_error,
            updated_at=now,
        )
        refreshed = self.store.get_state(runtime.service_name)
        if refreshed is None:
            raise RuntimeError(f"Missing service runtime row after failure for {runtime.service_name}")
        return refreshed

    def record_shutdown(self, *, runtime: ServiceRuntimeRecord, status: str) -> None:
        self.store.update_state(
            runtime.service_name,
            status=status,
            persona_url=None,
            persona_wait_started_at=None,
            persona_last_notification_at=None,
            last_heartbeat_at=datetime.now(UTC).isoformat(),
            updated_at=datetime.now(UTC).isoformat(),
        )
