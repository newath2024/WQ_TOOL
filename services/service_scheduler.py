from __future__ import annotations

from datetime import UTC, datetime

from core.config import ServiceConfig
from services.models import ServiceTickOutcome
from storage.models import ServiceRuntimeRecord


class ServiceScheduler:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._consecutive_empty_polls = 0

    def record_poll_result(self, new_result_count: int) -> None:
        if int(new_result_count) == 0:
            self._consecutive_empty_polls += 1
            return
        self._consecutive_empty_polls = 0

    def next_sleep_seconds(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        outcome: ServiceTickOutcome,
        now: str | None = None,
    ) -> int:
        del runtime
        if outcome.next_sleep_seconds > 0:
            return int(outcome.next_sleep_seconds)
        current = datetime.fromisoformat(now) if now else datetime.now(UTC)
        if outcome.cooldown_until:
            cooldown_until = datetime.fromisoformat(outcome.cooldown_until)
            remaining = max(int((cooldown_until - current).total_seconds()), 0)
            if remaining > 0:
                return max(1, min(remaining, self.config.heartbeat_interval_seconds))
        if outcome.status == "waiting_persona_confirmation":
            return self.config.persona_confirmation_poll_interval_seconds
        if outcome.status in {"waiting_persona", "auth_throttled"}:
            return self.config.persona_retry_interval_seconds
        if outcome.status == "paused_quarantine":
            return self.config.heartbeat_interval_seconds
        if outcome.pending_job_count > 0:
            base = int(self.config.poll_interval_seconds)
            multiplier = 2 ** min(self._consecutive_empty_polls, 3)
            interval = min(base * multiplier, 15)
            return int(interval)
        self._consecutive_empty_polls = 0
        if outcome.status in {"idle", "no_candidates"}:
            return self.config.idle_sleep_seconds
        return self.config.tick_interval_seconds
