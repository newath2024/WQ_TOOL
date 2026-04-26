from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from domain.exceptions import ConcurrentSimulationLimitExceeded
from services.models import ServiceTickOutcome
from storage.models import (
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
)

class CooldownPolicy:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _submission_cooldown_outcome(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        now: str,
        error: ConcurrentSimulationLimitExceeded,
        logger,
    ) -> ServiceTickOutcome:
        cooldown_seconds = max(int(error.cooldown_seconds), 1)
        cooldown_until = (datetime.fromisoformat(now) + timedelta(seconds=cooldown_seconds)).isoformat()
        pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
        pending_jobs = len(pending_submissions)
        observed_pending_limit = self._derive_observed_limit_telemetry(
            runtime=runtime,
            pending_jobs=pending_jobs,
        )
        self._persist_limit_hit_telemetry(
            runtime=runtime,
            observed_pending_limit=observed_pending_limit,
            cooldown_until=cooldown_until,
            last_error=str(error),
            updated_at=now,
        )
        logger.warning(
            "BRAIN concurrent simulation limit hit; pausing submissions for %ss pending_jobs=%s observed_pending_limit=%s",
            cooldown_seconds,
            pending_jobs,
            observed_pending_limit,
        )
        return self._defer_pending_timeouts_for_wait(
            run_id=runtime.service_run_id,
            updated_at=now,
            outcome=self._with_queue_metrics(
                ServiceTickOutcome(
                    status="cooldown",
                    pending_job_count=pending_jobs,
                    active_batch_id=pending_submissions[0].batch_id if pending_submissions else runtime.active_batch_id,
                    next_sleep_seconds=0 if pending_jobs > 0 else cooldown_seconds,
                    last_error=str(error),
                    cooldown_until=cooldown_until,
                ),
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
            ),
        )

    def _defer_pending_timeouts_for_wait(
        self,
        *,
        run_id: str,
        updated_at: str,
        outcome: ServiceTickOutcome,
    ) -> ServiceTickOutcome:
        seconds = float(outcome.next_sleep_seconds)
        if seconds <= 0 and outcome.cooldown_until:
            try:
                now = datetime.fromisoformat(updated_at)
                cooldown_until = datetime.fromisoformat(outcome.cooldown_until)
            except ValueError:
                cooldown_remaining = 0.0
            else:
                cooldown_remaining = max((cooldown_until - now).total_seconds(), 0.0)
            if cooldown_remaining > 0:
                seconds = min(cooldown_remaining, float(self.config.service.heartbeat_interval_seconds))
                if seconds > 0:
                    seconds = max(seconds, 1.0)
        if outcome.pending_job_count > 0 and seconds > 0:
            self.brain_service.defer_pending_timeouts(
                run_id=run_id,
                seconds=seconds,
                updated_at=updated_at,
            )
        return outcome

    @staticmethod
    def _is_dispatch_blocked_status(status: str) -> bool:
        return status in {
            "auth_throttled",
            "auth_unavailable",
            "cooldown",
            "waiting_persona",
            "waiting_persona_confirmation",
        }

    def _effective_pending_cap(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        now: str | None = None,
        allow_observed_limit_probe: bool = False,
    ) -> int:
        hard_cap = max(int(self.config.service.max_pending_jobs), 0)
        state = self._observed_limit_state(runtime=runtime, now=now, clear_expired=True)
        if not state:
            return hard_cap
        observed_limit = min(hard_cap, int(state["limit"]))
        if observed_limit >= hard_cap:
            return hard_cap
        if allow_observed_limit_probe or self._observed_limit_probe_due(state=state, now=now):
            return min(hard_cap, observed_limit + 1)
        return observed_limit

    def _observed_limit_state(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        now: str | None = None,
        clear_expired: bool = False,
    ) -> dict[str, object]:
        current = self.repository.service_runtime.get_state(runtime.service_name) or runtime
        counters = self._decode_json_object(current.counters_json)
        try:
            observed_limit = int(counters.get("observed_pending_limit") or 0)
        except (TypeError, ValueError):
            observed_limit = 0
        observed_at = str(counters.get("observed_limit_observed_at") or "")
        if observed_limit <= 0 or not observed_at:
            return {}
        now_dt = self._parse_datetime(now)
        observed_at_dt = self._parse_datetime(observed_at)
        if now_dt is not None and observed_at_dt is not None:
            age_seconds = (now_dt - observed_at_dt).total_seconds()
            if age_seconds > float(self.config.service.observed_limit_ttl_seconds):
                if clear_expired:
                    self._clear_observed_limit(runtime=runtime, updated_at=now or datetime.now(UTC).isoformat())
                return {}
        return {
            "limit": observed_limit,
            "observed_at": observed_at,
            "last_probe_at": str(counters.get("observed_limit_last_probe_at") or ""),
        }

    def _observed_limit_probe_due(self, *, state: dict[str, object], now: str | None = None) -> bool:
        last_probe_at = str(state.get("last_probe_at") or "")
        if not last_probe_at:
            return True
        now_dt = self._parse_datetime(now)
        probe_dt = self._parse_datetime(last_probe_at)
        if now_dt is None or probe_dt is None:
            return True
        return (now_dt - probe_dt).total_seconds() >= float(self.config.service.observed_limit_probe_interval_seconds)

    def _mark_observed_limit_probe_if_needed(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        pending_jobs: int,
        updated_at: str,
    ) -> bool:
        state = self._observed_limit_state(runtime=runtime, now=updated_at)
        if not state:
            return False
        observed_limit = int(state["limit"])
        if pending_jobs < observed_limit:
            return False
        current = self.repository.service_runtime.get_state(runtime.service_name)
        if current is None:
            return True
        counters = self._decode_json_object(current.counters_json)
        counters["observed_limit_last_probe_at"] = updated_at
        self.repository.service_runtime.update_state(
            runtime.service_name,
            counters_json=json.dumps(counters, sort_keys=True),
            updated_at=updated_at,
        )
        return True

    def _clear_observed_limit(self, *, runtime: ServiceRuntimeRecord, updated_at: str) -> None:
        current = self.repository.service_runtime.get_state(runtime.service_name)
        if current is None:
            return
        counters = self._decode_json_object(current.counters_json)
        for key in (
            "observed_pending_limit",
            "observed_limit_observed_at",
            "observed_limit_last_probe_at",
        ):
            counters.pop(key, None)
        self.repository.service_runtime.update_state(
            runtime.service_name,
            counters_json=json.dumps(counters, sort_keys=True),
            updated_at=updated_at,
        )

    def _derive_observed_limit_telemetry(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        pending_jobs: int,
    ) -> int:
        partial_batch = self._latest_interrupted_limit_batch(run_id=runtime.service_run_id)
        partial_count = 0
        if partial_batch is not None:
            partial_count = self._partial_submitted_count(batch=partial_batch)
        return max(int(pending_jobs), int(partial_count), 0)

    def _latest_interrupted_limit_batch(self, *, run_id: str) -> SubmissionBatchRecord | None:
        candidates = [
            batch
            for batch in self.repository.submissions.list_batches(run_id)
            if str(batch.service_status_reason or "") == "submission_failed:ConcurrentSimulationLimitExceeded"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda batch: (batch.updated_at, batch.created_at, batch.batch_id))

    def _partial_submitted_count(self, *, batch: SubmissionBatchRecord) -> int:
        notes = self._decode_json_object(batch.notes_json)
        submitted_count = notes.get("submitted_candidate_count")
        try:
            parsed = int(submitted_count)
        except (TypeError, ValueError):
            parsed = int(batch.candidate_count or 0)
        return max(parsed, 0)

    def _persist_limit_hit_telemetry(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        observed_pending_limit: int,
        cooldown_until: str,
        last_error: str,
        updated_at: str,
    ) -> None:
        current = self.repository.service_runtime.get_state(runtime.service_name)
        if current is None:
            return
        counters = self._decode_json_object(current.counters_json)
        try:
            hit_count = int(counters.get("observed_limit_hit_count") or 0)
        except (TypeError, ValueError):
            hit_count = 0
        counters["learned_safe_cap"] = observed_pending_limit
        counters["observed_pending_limit"] = observed_pending_limit
        counters["observed_limit_observed_at"] = updated_at
        counters["observed_limit_last_probe_at"] = updated_at
        counters["observed_limit_hit_count"] = hit_count + 1
        self.repository.service_runtime.update_state(
            runtime.service_name,
            counters_json=json.dumps(counters, sort_keys=True),
            cooldown_until=cooldown_until,
            last_error=last_error,
            updated_at=updated_at,
        )

