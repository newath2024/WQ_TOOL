from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime

from core.logging import get_logger
from services.models import ServiceTickOutcome
from services.progress_log import append_progress_event
from storage.models import (
    ServiceDispatchQueueRecord,
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
)

class QueueManager:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _prepare_and_refill_dispatch_queue(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        pending_cap: int | None = None,
        now: str | None = None,
    ) -> ServiceTickOutcome:
        now = now or datetime.now(UTC).isoformat()
        effective_pending_cap = max(int(pending_cap or self.config.service.max_pending_jobs), 0)
        pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
        initial_queue_depth = self._queue_depth(service_name=runtime.service_name, run_id=runtime.service_run_id)
        aggregate = self._with_queue_metrics(
            ServiceTickOutcome(
                status="running" if pending_submissions or initial_queue_depth > 0 else "idle",
                pending_job_count=len(pending_submissions),
                active_batch_id=(pending_submissions[0].batch_id if pending_submissions else runtime.active_batch_id),
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )
        if effective_pending_cap <= 0:
            return aggregate
        max_iterations = max(effective_pending_cap, 1) + 4
        for _ in range(max_iterations):
            if aggregate.queue_depth <= 0:
                pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
                if pending_submissions:
                    append_progress_event(
                        self.config,
                        self.environment,
                        event="queue_prepare_deferred",
                        stage="service-queue-prepare",
                        status="deferred",
                        tick_id=tick_id,
                        batch_id=pending_submissions[0].batch_id,
                        payload={
                            "pending_job_count": len(pending_submissions),
                            "queue_depth": aggregate.queue_depth,
                            "max_pending_jobs": self.config.service.max_pending_jobs,
                            "reason": "pending_jobs_active",
                        },
                    )
                    return self._with_queue_metrics(
                        replace(
                            aggregate,
                            status="running",
                            pending_job_count=len(pending_submissions),
                            active_batch_id=pending_submissions[0].batch_id,
                        ),
                        service_name=runtime.service_name,
                        run_id=runtime.service_run_id,
                    )
                prepare_outcome = self._prepare_dispatch_queue_if_needed(runtime=runtime, tick_id=tick_id)
                aggregate = self._merge_tick_outcomes(
                    poll_outcome=aggregate,
                    submit_outcome=prepare_outcome,
                )
                if self._is_dispatch_blocked_status(prepare_outcome.status) or prepare_outcome.status == "no_candidates":
                    return aggregate
                if aggregate.queue_depth <= 0:
                    return aggregate

            pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            if len(pending_submissions) >= effective_pending_cap:
                return self._with_queue_metrics(
                    replace(
                        aggregate,
                        status="running" if pending_submissions or aggregate.queue_depth > 0 else "idle",
                        pending_job_count=len(pending_submissions),
                        active_batch_id=(
                            pending_submissions[0].batch_id if pending_submissions else aggregate.active_batch_id
                        ),
                    ),
                    service_name=runtime.service_name,
                    run_id=runtime.service_run_id,
                )

            dispatch_outcome = self._dispatch_queued_candidates(
                runtime=runtime,
                tick_id=tick_id,
                pending_cap=effective_pending_cap,
                now=now,
            )
            aggregate = self._merge_tick_outcomes(
                poll_outcome=aggregate,
                submit_outcome=dispatch_outcome,
            )
            if self._is_dispatch_blocked_status(dispatch_outcome.status):
                return aggregate
            if dispatch_outcome.submitted_count == 0 and aggregate.queue_depth <= 0:
                return aggregate

        return aggregate

    def _prepare_dispatch_queue_if_needed(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
    ) -> ServiceTickOutcome:
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-queue-prepare",
            tick_id=tick_id,
        )
        if self._queue_depth(service_name=runtime.service_name, run_id=runtime.service_run_id) > 0:
            pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            return self._with_queue_metrics(
                ServiceTickOutcome(
                    status="running",
                    pending_job_count=len(pending_submissions),
                    active_batch_id=(pending_submissions[0].batch_id if pending_submissions else runtime.active_batch_id),
                ),
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
            )

        blocked_outcome = self._probe_auth_after_batch_failures(runtime=runtime, logger=logger)
        if blocked_outcome is not None:
            return blocked_outcome

        pre_prepare_pending_job_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        source_round_index = self._next_service_round_index(run_id=runtime.service_run_id)
        mutation_parent_ids = self._select_mutation_parent_ids(run_id=runtime.service_run_id)
        prepare_started = time.perf_counter()
        batch_result = self.batch_service.prepare_service_batch(
            config=self.config,
            environment=self.environment,
            count=self.config.loop.generation_batch_size,
            mutation_parent_ids=mutation_parent_ids,
            round_index=source_round_index,
        )
        prepare_batch_ms = (time.perf_counter() - prepare_started) * 1000.0
        candidates = list(batch_result.candidates)
        if candidates:
            self.repository.save_alpha_candidates(runtime.service_run_id, candidates)
        selected_scores = list(batch_result.selected[: self.config.loop.simulation_batch_size])
        selected_candidates = [item.candidate for item in selected_scores]
        logger.info("[generation-summary] %s", json.dumps(batch_result.generation_stage_metrics, sort_keys=True))
        self._sync_service_round(
            run_id=runtime.service_run_id,
            round_index_override=source_round_index,
            generated_count=len(candidates),
            validated_count=batch_result.validated_count or len(candidates),
            mutated_children_count=batch_result.mutated_children_count,
            status_override="queued" if selected_candidates else "no_candidates",
            note_overrides={
                "dispatch_mode": "service_single",
                "archived_count": batch_result.archived_count,
                "generation_stage_metrics": batch_result.generation_stage_metrics,
                "prepared_queue_size": len(selected_candidates),
            },
        )
        if not selected_candidates:
            logger.info(
                "No simulation-worthy candidates generated for new dispatch queue. top_fail_reasons=%s",
                batch_result.generation_stage_metrics.get("top_fail_reasons", {}),
            )
            pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            return self._with_queue_metrics(
                ServiceTickOutcome(
                    status="no_candidates",
                    pending_job_count=len(pending_submissions),
                    active_batch_id=(pending_submissions[0].batch_id if pending_submissions else runtime.active_batch_id),
                    generated_count=len(candidates),
                    prepare_batch_ms=prepare_batch_ms,
                    pre_prepare_pending_job_count=pre_prepare_pending_job_count,
                ),
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
            )

        timestamp = datetime.now(UTC).isoformat()
        next_position = self.repository.service_dispatch_queue.next_queue_position(
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )
        self.repository.service_dispatch_queue.upsert_items(
            [
                ServiceDispatchQueueRecord(
                    queue_item_id=self._queue_item_id(
                        run_id=runtime.service_run_id,
                        source_round_index=source_round_index,
                        queue_position=next_position + offset,
                    ),
                    service_name=runtime.service_name,
                    run_id=runtime.service_run_id,
                    candidate_id=candidate.alpha_id,
                    source_round_index=source_round_index,
                    queue_position=next_position + offset,
                    status="queued",
                    batch_id=None,
                    job_id=None,
                    failure_reason=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                for offset, candidate in enumerate(selected_candidates)
            ]
        )
        pending_submissions = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
        queue_depth = self._queue_depth(service_name=runtime.service_name, run_id=runtime.service_run_id)
        logger.info(
            "Prepared dispatch queue round=%s selected=%s queue_depth=%s pending=%s",
            source_round_index,
            len(selected_candidates),
            queue_depth,
            len(pending_submissions),
        )
        append_progress_event(
            self.config,
            self.environment,
            event="dispatch_queue_prepared",
            stage="service-queue-prepare",
            status="queued",
            tick_id=tick_id,
            round_index=source_round_index,
            payload={
                "generated_count": len(candidates),
                "prepared_queue_size": len(selected_candidates),
                "pending_job_count": len(pending_submissions),
                "pre_prepare_pending_job_count": pre_prepare_pending_job_count,
                "queue_depth": queue_depth,
            },
        )
        return self._with_queue_metrics(
            ServiceTickOutcome(
                status="running" if pending_submissions or queue_depth > 0 else "idle",
                pending_job_count=len(pending_submissions),
                active_batch_id=(pending_submissions[0].batch_id if pending_submissions else runtime.active_batch_id),
                generated_count=len(candidates),
                prepare_batch_ms=prepare_batch_ms,
                pre_prepare_pending_job_count=pre_prepare_pending_job_count,
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

    def _recover_dispatch_queue_items(
        self,
        *,
        service_name: str,
        run_id: str,
        now: str,
    ) -> tuple[int, int]:
        dispatching_items = self.repository.service_dispatch_queue.list_items(
            service_name=service_name,
            run_id=run_id,
            statuses=("dispatching",),
        )
        if not dispatching_items:
            return 0, 0

        batches_by_queue_item: dict[str, SubmissionBatchRecord] = {}
        for batch in self.repository.submissions.list_batches(run_id):
            queue_item_id = str(self._decode_json_object(batch.notes_json).get("dispatch_queue_item_id") or "").strip()
            if queue_item_id:
                batches_by_queue_item[queue_item_id] = batch

        active_submissions = self._active_submissions_by_candidate(run_id=run_id, excluding_batch_id="")
        terminal_results = self._terminal_results_by_candidate(run_id=run_id)
        recovered = 0
        requeued = 0
        for item in dispatching_items:
            submission = self.repository.submissions.get_submission(item.job_id) if item.job_id else None
            batch = self.repository.submissions.get_batch(item.batch_id) if item.batch_id else None
            if submission is None and item.queue_item_id in batches_by_queue_item:
                batch = batches_by_queue_item[item.queue_item_id]
                submissions = self.repository.submissions.list_submissions(run_id=run_id, batch_id=batch.batch_id)
                submission = submissions[0] if submissions else None
            if submission is None:
                submission = active_submissions.get(item.candidate_id)
            if submission is not None:
                self.repository.service_dispatch_queue.update_item(
                    item.queue_item_id,
                    status="submitted",
                    batch_id=submission.batch_id,
                    job_id=submission.job_id,
                    failure_reason=None,
                    updated_at=now,
                )
                recovered += 1
                continue
            terminal_result = terminal_results.get(item.candidate_id)
            if terminal_result is not None:
                self.repository.service_dispatch_queue.update_item(
                    item.queue_item_id,
                    status="submitted",
                    batch_id=terminal_result.batch_id,
                    job_id=terminal_result.job_id,
                    failure_reason=None,
                    updated_at=now,
                )
                recovered += 1
                continue
            self.repository.service_dispatch_queue.update_item(
                item.queue_item_id,
                status="queued",
                batch_id=None,
                job_id=None,
                failure_reason="dispatch_recovered_to_queue",
                updated_at=now,
            )
            requeued += 1
        return recovered, requeued

    @staticmethod
    def _queue_item_id(*, run_id: str, source_round_index: int, queue_position: int) -> str:
        return f"queue-{run_id[:8]}-r{int(source_round_index):06d}-{int(queue_position):08d}"

    def _next_service_round_index(self, *, run_id: str) -> int:
        row = self.repository.connection.execute(
            """
            SELECT MAX(round_index) AS max_round
            FROM (
                SELECT COALESCE(MAX(round_index), 0) AS round_index
                FROM closed_loop_rounds
                WHERE run_id = ?
                UNION ALL
                SELECT COALESCE(MAX(round_index), 0) AS round_index
                FROM submission_batches
                WHERE run_id = ?
            )
            """,
            (run_id, run_id),
        ).fetchone()
        return int(row["max_round"] or 0) + 1 if row is not None else 1

