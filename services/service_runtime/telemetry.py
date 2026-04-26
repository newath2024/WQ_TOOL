from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime

from services.models import ServiceTickOutcome

class Telemetry:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    @staticmethod
    def _parse_datetime(timestamp: str | None) -> datetime | None:
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None

    @staticmethod
    def _decode_json_object(payload: str | None) -> dict[str, object]:
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _merge_tick_outcomes(
        *,
        poll_outcome: ServiceTickOutcome,
        submit_outcome: ServiceTickOutcome,
    ) -> ServiceTickOutcome:
        status = submit_outcome.status
        if status == "no_candidates" and submit_outcome.pending_job_count > 0:
            status = "running"
        return ServiceTickOutcome(
            status=status,
            pending_job_count=submit_outcome.pending_job_count,
            new_result_count=poll_outcome.new_result_count + submit_outcome.new_result_count,
            active_batch_id=submit_outcome.active_batch_id or poll_outcome.active_batch_id,
            queue_depth=submit_outcome.queue_depth,
            queue_counts=dict(submit_outcome.queue_counts),
            next_sleep_seconds=submit_outcome.next_sleep_seconds,
            generated_count=poll_outcome.generated_count + submit_outcome.generated_count,
            submitted_count=poll_outcome.submitted_count + submit_outcome.submitted_count,
            completed_count=poll_outcome.completed_count + submit_outcome.completed_count,
            failed_count=poll_outcome.failed_count + submit_outcome.failed_count,
            quarantined_count=poll_outcome.quarantined_count + submit_outcome.quarantined_count,
            last_error=submit_outcome.last_error or poll_outcome.last_error,
            persona_url=submit_outcome.persona_url or poll_outcome.persona_url,
            cooldown_until=submit_outcome.cooldown_until or poll_outcome.cooldown_until,
            poll_pending_ms=poll_outcome.poll_pending_ms + submit_outcome.poll_pending_ms,
            prepare_batch_ms=poll_outcome.prepare_batch_ms + submit_outcome.prepare_batch_ms,
            submit_batch_ms=poll_outcome.submit_batch_ms + submit_outcome.submit_batch_ms,
            pre_prepare_pending_job_count=(
                submit_outcome.pre_prepare_pending_job_count
                if submit_outcome.pre_prepare_pending_job_count is not None
                else poll_outcome.pre_prepare_pending_job_count
            ),
        )

    def _with_queue_metrics(
        self,
        outcome: ServiceTickOutcome,
        *,
        service_name: str,
        run_id: str,
    ) -> ServiceTickOutcome:
        queue_counts = self._queue_counts(service_name=service_name, run_id=run_id)
        return replace(
            outcome,
            queue_depth=int(queue_counts.get("dispatching", 0) + queue_counts.get("queued", 0)),
            queue_counts=queue_counts,
        )

    def _queue_counts(
        self,
        *,
        service_name: str,
        run_id: str,
        source_round_index: int | None = None,
    ) -> dict[str, int]:
        counts = Counter(
            item.status
            for item in self.repository.service_dispatch_queue.list_items(
                service_name=service_name,
                run_id=run_id,
                source_round_index=source_round_index,
            )
            if item.status
        )
        return {key: counts[key] for key in sorted(counts)}

    def _queue_depth(self, *, service_name: str, run_id: str) -> int:
        queue_counts = self._queue_counts(service_name=service_name, run_id=run_id)
        return int(queue_counts.get("dispatching", 0) + queue_counts.get("queued", 0))

    def _recent_job_failure_rate(self, run_id: str, *, lookback_batches: int = 10) -> float:
        completed_batches = self.repository.submissions.list_batches_by_status(
            run_id=run_id,
            statuses=("completed",),
        )
        if not completed_batches:
            return 0.0
        latest_batches = sorted(
            completed_batches,
            key=lambda item: (item.created_at, item.batch_id),
            reverse=True,
        )[:lookback_batches]
        failed_jobs = 0
        total_jobs = 0
        for batch in latest_batches:
            submissions = self.repository.submissions.list_submissions(
                run_id=run_id,
                batch_id=batch.batch_id,
            )
            if not submissions:
                continue
            total_jobs += len(submissions)
            failed_jobs += sum(
                1
                for submission in submissions
                if submission.status in {"failed", "rejected", "timeout"}
            )
        if total_jobs <= 0:
            return 0.0
        return float(failed_jobs) / float(total_jobs)

    def _recent_live_timeout_rate(self, run_id: str, *, lookback_batches: int = 10) -> float:
        completed_batches = self.repository.submissions.list_batches_by_status(
            run_id=run_id,
            statuses=("completed",),
        )
        if not completed_batches:
            return 0.0
        latest_batches = sorted(
            completed_batches,
            key=lambda item: (item.created_at, item.batch_id),
            reverse=True,
        )[:lookback_batches]
        live_timeout_jobs = 0
        total_jobs = 0
        for batch in latest_batches:
            submissions = self.repository.submissions.list_submissions(
                run_id=run_id,
                batch_id=batch.batch_id,
            )
            if not submissions:
                continue
            total_jobs += len(submissions)
            live_timeout_jobs += sum(
                1
                for submission in submissions
                if submission.status == "timeout"
                and str(submission.service_failure_reason or submission.error_message or "") == "poll_timeout_live"
            )
        if total_jobs <= 0:
            return 0.0
        return float(live_timeout_jobs) / float(total_jobs)

