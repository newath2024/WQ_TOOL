from __future__ import annotations

import time
from datetime import UTC, datetime

from domain.exceptions import ConcurrentSimulationLimitExceeded
from core.logging import get_logger
from services.models import ServiceTickOutcome
from services.progress_log import append_progress_event
from storage.models import (
    ServiceRuntimeRecord,
)

class BatchSubmitter:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _submit_new_batch(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        pending_cap: int | None = None,
    ) -> ServiceTickOutcome:
        return self._prepare_and_refill_dispatch_queue(
            runtime=runtime,
            tick_id=tick_id,
            pending_cap=pending_cap,
            now=datetime.now(UTC).isoformat(),
        )

    def _dispatch_queued_candidates(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        pending_cap: int,
        now: str | None = None,
    ) -> ServiceTickOutcome:
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-dispatch",
            tick_id=tick_id,
        )
        submit_started = time.perf_counter()
        submitted_count = 0
        active_batch_id = runtime.active_batch_id
        current_pending_cap = int(pending_cap)
        now = now or datetime.now(UTC).isoformat()

        while True:
            pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            available_slots = max(current_pending_cap - len(pending_submissions), 0)
            if available_slots <= 0:
                break
            queued_items = self.repository.service_dispatch_queue.list_items(
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
                statuses=("queued",),
            )
            if not queued_items:
                break

            queue_item = queued_items[0]
            now = datetime.now(UTC).isoformat()
            self.repository.service_dispatch_queue.update_item(
                queue_item.queue_item_id,
                status="dispatching",
                failure_reason=None,
                updated_at=now,
            )
            active_submission = self._active_submissions_by_candidate(
                run_id=runtime.service_run_id,
                excluding_batch_id="",
            ).get(queue_item.candidate_id)
            if active_submission is not None:
                self.repository.service_dispatch_queue.update_item(
                    queue_item.queue_item_id,
                    status="submitted",
                    batch_id=active_submission.batch_id,
                    job_id=active_submission.job_id,
                    failure_reason=None,
                    updated_at=now,
                )
                self._sync_service_round(
                    run_id=runtime.service_run_id,
                    batch_id=active_submission.batch_id,
                )
                continue

            terminal_result = self._terminal_results_by_candidate(run_id=runtime.service_run_id).get(queue_item.candidate_id)
            if terminal_result is not None:
                self.repository.service_dispatch_queue.update_item(
                    queue_item.queue_item_id,
                    status="submitted",
                    batch_id=terminal_result.batch_id,
                    job_id=terminal_result.job_id,
                    failure_reason=None,
                    updated_at=now,
                )
                self._sync_service_round(
                    run_id=runtime.service_run_id,
                    batch_id=terminal_result.batch_id,
                )
                continue

            candidate = self._load_candidates_by_ids(
                run_id=runtime.service_run_id,
                candidate_ids={queue_item.candidate_id},
            ).get(queue_item.candidate_id)
            if candidate is None:
                self.repository.service_dispatch_queue.update_item(
                    queue_item.queue_item_id,
                    status="dropped",
                    batch_id=None,
                    job_id=None,
                    failure_reason="dispatch_candidate_missing",
                    updated_at=now,
                )
                continue

            probing_observed_limit = self._mark_observed_limit_probe_if_needed(
                runtime=runtime,
                pending_jobs=len(pending_submissions),
                updated_at=now,
            )
            try:
                batch = self.brain_service.submit_candidates(
                    [candidate],
                    config=self.config,
                    environment=self.environment,
                    round_index=queue_item.source_round_index,
                    batch_size=1,
                    note_overrides={
                        "dispatch_mode": "service_single",
                        "dispatch_queue_item_id": queue_item.queue_item_id,
                        "source_round_index": queue_item.source_round_index,
                    },
                )
            except ConcurrentSimulationLimitExceeded:
                self.repository.service_dispatch_queue.update_item(
                    queue_item.queue_item_id,
                    status="queued",
                    batch_id=None,
                    job_id=None,
                    failure_reason="dispatch_concurrent_limit",
                    updated_at=datetime.now(UTC).isoformat(),
                )
                raise
            except Exception:
                self.repository.service_dispatch_queue.update_item(
                    queue_item.queue_item_id,
                    status="queued",
                    batch_id=None,
                    job_id=None,
                    failure_reason="dispatch_interrupted",
                    updated_at=datetime.now(UTC).isoformat(),
                )
                raise

            batch_job_id = batch.jobs[0].job_id if batch.jobs else None
            if probing_observed_limit:
                self._clear_observed_limit(runtime=runtime, updated_at=datetime.now(UTC).isoformat())
                current_pending_cap = max(int(self.config.service.max_pending_jobs), 0)
            self.repository.service_dispatch_queue.update_item(
                queue_item.queue_item_id,
                status="submitted",
                batch_id=batch.batch_id,
                job_id=batch_job_id,
                failure_reason=None,
                updated_at=datetime.now(UTC).isoformat(),
            )
            self._sync_service_round(
                run_id=runtime.service_run_id,
                batch_id=batch.batch_id,
                status_override=batch.status,
                note_overrides={
                    "dispatch_mode": "service_single",
                    "source_round_index": queue_item.source_round_index,
                },
            )
            pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            active_batch_id = batch.batch_id
            submitted_count += 1
            logger.info(
                "Dispatched queue_item=%s candidate=%s batch=%s pending=%s",
                queue_item.queue_item_id,
                queue_item.candidate_id,
                batch.batch_id,
                pending_count,
            )
            append_progress_event(
                self.config,
                self.environment,
                event="batch_submitted",
                stage="service-dispatch",
                status=batch.status,
                tick_id=tick_id,
                round_index=queue_item.source_round_index,
                batch_id=batch.batch_id,
                payload={
                    "generated_count": 0,
                    "submitted_count": 1,
                    "pending_job_count": pending_count,
                    "export_path": batch.export_path,
                    "dispatch_queue_item_id": queue_item.queue_item_id,
                    "queue_depth": self._queue_depth(service_name=runtime.service_name, run_id=runtime.service_run_id),
                },
            )

        pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
        next_active_batch_id = pending_submissions[0].batch_id if pending_submissions else active_batch_id
        return self._with_queue_metrics(
            ServiceTickOutcome(
                status="running" if pending_submissions or self._queue_depth(service_name=runtime.service_name, run_id=runtime.service_run_id) > 0 else "idle",
                pending_job_count=len(pending_submissions),
                active_batch_id=next_active_batch_id,
                submitted_count=submitted_count,
                submit_batch_ms=(time.perf_counter() - submit_started) * 1000.0,
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

