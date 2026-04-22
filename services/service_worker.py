from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from adapters.brain_api_adapter import ConcurrentSimulationLimitExceeded
from core.config import AppConfig
from core.logging import get_logger
from services.brain_batch_service import BrainBatchService
from services.brain_learning_service import BrainLearningService
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import load_research_context
from services.evaluation_service import alpha_candidate_from_record
from generator.engine import AlphaCandidate
from services.models import CommandEnvironment, ServiceTickOutcome, SimulationResult
from services.notification_manager import NotificationManager
from services.progress_log import append_progress_event
from services.session_manager import SessionManager
from storage.models import (
    BrainResultRecord,
    ClosedLoopRoundRecord,
    ServiceDispatchQueueRecord,
    ServiceRuntimeRecord,
    SubmissionBatchRecord,
    SubmissionRecord,
)
from storage.repository import SQLiteRepository


class ServiceWorker:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        brain_service: BrainService,
        batch_service: BrainBatchService | None = None,
        selection_service: CandidateSelectionService | None = None,
        learning_service: BrainLearningService | None = None,
        session_manager: SessionManager,
        notification_manager: NotificationManager,
    ) -> None:
        self.repository = repository
        self.config = config
        self.environment = environment
        self.brain_service = brain_service
        self.selection_service = selection_service or CandidateSelectionService(
            repository=repository,
            adaptive_config=config.adaptive_generation,
        )
        self.selection_service.configure_runtime(
            repository=repository,
            adaptive_config=config.adaptive_generation,
        )
        self.batch_service = batch_service or BrainBatchService(repository, selection_service=self.selection_service)
        self.learning_service = learning_service or BrainLearningService(
            repository,
            memory_service=self.selection_service.memory_service,
        )
        self.session_manager = session_manager
        self.notification_manager = notification_manager

    def run_tick(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        stop_requested: bool = False,
    ) -> ServiceTickOutcome:
        now = datetime.now(UTC).isoformat()
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-tick",
            tick_id=tick_id,
        )
        should_probe_auth = self._should_probe_session_during_cooldown(runtime)
        submission_cooldown_active = bool(runtime.cooldown_until and runtime.cooldown_until > now and not should_probe_auth)
        if submission_cooldown_active:
            logger.info(
                "Service is cooling down until %s; polling existing work but skipping new submissions.",
                runtime.cooldown_until,
            )
        if runtime.cooldown_until and runtime.cooldown_until > now and should_probe_auth:
            logger.info("Auth-related cooldown detected; probing session state before waiting out cooldown.")

        session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=False)
        if session_state.status == "authentication_required":
            confirmation = self.notification_manager.request_persona_confirmation(
                runtime=runtime,
                service_name=runtime.service_name,
            )
            self._update_persona_confirmation_state(
                service_name=runtime.service_name,
                now=now,
                nonce=confirmation.nonce,
                last_prompt_at=confirmation.last_prompt_at,
                granted_at=confirmation.granted_at,
                last_update_id=confirmation.last_update_id,
            )
            runtime = self.repository.service_runtime.get_state(runtime.service_name) or runtime
            if confirmation.status == "pending":
                pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
                logger.warning(
                    "Waiting for Telegram confirmation before requesting Persona link; pending_jobs=%s",
                    pending_jobs,
                )
                return self._defer_pending_timeouts_for_wait(
                    run_id=runtime.service_run_id,
                    updated_at=now,
                    outcome=ServiceTickOutcome(
                        status="waiting_persona_confirmation",
                        pending_job_count=pending_jobs,
                        active_batch_id=runtime.active_batch_id,
                        persona_url=None,
                        last_error=confirmation.detail,
                        next_sleep_seconds=self.config.service.persona_confirmation_poll_interval_seconds,
                    ),
                )
            session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=True)
        if session_state.status == "waiting_persona":
            if self.notification_manager.requires_fresh_persona_confirmation_for_url(
                runtime=runtime,
                persona_url=str(session_state.persona_url or ""),
                now=now,
            ):
                confirmation = self.notification_manager.request_persona_confirmation(
                    runtime=runtime,
                    service_name=runtime.service_name,
                )
                self._update_persona_confirmation_state(
                    service_name=runtime.service_name,
                    now=now,
                    nonce=confirmation.nonce,
                    last_prompt_at=confirmation.last_prompt_at,
                    granted_at=confirmation.granted_at,
                    last_update_id=confirmation.last_update_id,
                )
                runtime = self.repository.service_runtime.get_state(runtime.service_name) or runtime
                if confirmation.status == "pending":
                    pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
                    logger.warning(
                        "Waiting for Telegram confirmation before delivering Persona link; pending_jobs=%s",
                        pending_jobs,
                    )
                    return self._defer_pending_timeouts_for_wait(
                        run_id=runtime.service_run_id,
                        updated_at=now,
                        outcome=ServiceTickOutcome(
                            status="waiting_persona_confirmation",
                            pending_job_count=pending_jobs,
                            active_batch_id=runtime.active_batch_id,
                            persona_url=None,
                            last_error=confirmation.detail,
                            next_sleep_seconds=self.config.service.persona_confirmation_poll_interval_seconds,
                        ),
                    )
            persona_wait_started_at = runtime.persona_wait_started_at or now
            if session_state.persona_url and session_state.persona_url != runtime.persona_url:
                persona_wait_started_at = now
            sent, notification_at = self.notification_manager.notify_persona_required(
                runtime=runtime,
                persona_url=str(session_state.persona_url or ""),
            )
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="waiting_persona",
                persona_url=session_state.persona_url,
                persona_wait_started_at=persona_wait_started_at,
                persona_last_notification_at=notification_at if sent else runtime.persona_last_notification_at,
                persona_confirmation_nonce=None,
                persona_confirmation_last_prompt_at=None,
                persona_confirmation_granted_at=None,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning("Persona verification required; pending_jobs=%s", pending_jobs)
            return self._defer_pending_timeouts_for_wait(
                run_id=runtime.service_run_id,
                updated_at=now,
                outcome=ServiceTickOutcome(
                    status="waiting_persona",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    persona_url=session_state.persona_url,
                    last_error=session_state.detail,
                    next_sleep_seconds=(
                        session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
                    ),
                ),
            )
        if session_state.status == "auth_throttled":
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="auth_throttled",
                persona_url=None,
                persona_wait_started_at=runtime.persona_wait_started_at or now,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning(
                "BRAIN biometrics throttled; pending_jobs=%s retry_after=%s",
                pending_jobs,
                session_state.retry_after_seconds,
            )
            return self._defer_pending_timeouts_for_wait(
                run_id=runtime.service_run_id,
                updated_at=now,
                outcome=ServiceTickOutcome(
                    status="auth_throttled",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    persona_url=None,
                    last_error=session_state.detail,
                    next_sleep_seconds=(
                        session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
                    ),
                ),
            )
        if session_state.status == "auth_unavailable":
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="auth_unavailable",
                persona_url=None,
                persona_wait_started_at=runtime.persona_wait_started_at,
                last_error=session_state.detail,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning(
                "BRAIN auth probe failed; pending_jobs=%s detail=%s",
                pending_jobs,
                session_state.detail,
            )
            return self._defer_pending_timeouts_for_wait(
                run_id=runtime.service_run_id,
                updated_at=now,
                outcome=ServiceTickOutcome(
                    status="auth_unavailable",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    persona_url=None,
                    last_error=session_state.detail,
                    next_sleep_seconds=self.config.service.tick_interval_seconds,
                ),
            )

        if runtime.cooldown_until and runtime.cooldown_until > now and should_probe_auth:
            logger.info("Authentication recovered during cooldown; resuming normal service work.")
        if (
            runtime.persona_url
            or runtime.persona_wait_started_at
            or runtime.persona_confirmation_nonce
            or runtime.persona_confirmation_granted_at
        ):
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="running",
                persona_url=None,
                persona_wait_started_at=None,
                persona_last_notification_at=None,
                persona_confirmation_nonce=None,
                persona_confirmation_last_prompt_at=None,
                persona_confirmation_granted_at=None,
                persona_confirmation_last_update_id=None,
                updated_at=now,
            )

        try:
            recovered_batch_ids, failed_batch_ids, resubmitted_batch_ids, quarantined_batch_ids = self._recover_submitting_batches(
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
                now=now,
                allow_resubmit=not submission_cooldown_active,
            )
        except ConcurrentSimulationLimitExceeded as exc:
            return self._submission_cooldown_outcome(runtime=runtime, now=now, error=exc, logger=logger)
        if recovered_batch_ids:
            logger.info("Recovered %s interrupted submitting batches", len(recovered_batch_ids))
        if failed_batch_ids:
            logger.warning("Marked %s ambiguous batches as failed to unblock service.", len(failed_batch_ids))
        if resubmitted_batch_ids:
            logger.warning("Resubmitted %s ambiguous batches after interrupted submission.", len(resubmitted_batch_ids))
        recovered_queue_items, requeued_queue_items = self._recover_dispatch_queue_items(
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
            now=now,
        )
        if recovered_queue_items:
            logger.info("Recovered %s dispatch queue items with existing submissions.", recovered_queue_items)
        if requeued_queue_items:
            logger.info("Re-queued %s interrupted dispatch queue items.", requeued_queue_items)
        paused_batches = self.repository.submissions.list_batches_by_status(
            run_id=runtime.service_run_id,
            statuses=("paused_quarantine",),
        )
        if submission_cooldown_active:
            paused_batches = [batch for batch in paused_batches if not self._is_ambiguous_batch(batch)]
        if paused_batches:
            active_batch_id = paused_batches[0].batch_id
            logger.error("Service remains paused because batch=%s is quarantined.", active_batch_id)
            return self._with_queue_metrics(
                ServiceTickOutcome(
                    status="paused_quarantine",
                    pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                    active_batch_id=active_batch_id,
                    quarantined_count=len(quarantined_batch_ids),
                ),
                service_name=runtime.service_name,
                run_id=runtime.service_run_id,
            )

        pending_batches = self.repository.submissions.list_batches_by_status(
            run_id=runtime.service_run_id,
            statuses=("submitting", "submitted", "running"),
        )
        poll_outcome = self._with_queue_metrics(
            ServiceTickOutcome(
                status="idle",
                pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                active_batch_id=runtime.active_batch_id,
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )
        if pending_batches:
            poll_outcome = self._poll_pending_batches(
                runtime=runtime,
                tick_id=tick_id,
                batch_ids=[batch.batch_id for batch in pending_batches],
            )
        reconciled_batches = self._reconcile_completed_batches(run_id=runtime.service_run_id)
        if reconciled_batches:
            logger.info("Reconciled learning for %s completed batches after polling.", reconciled_batches)

        if stop_requested:
            logger.info("Shutdown requested after polling; skipping top-up submission.")
            return poll_outcome
        if submission_cooldown_active:
            if poll_outcome.new_result_count > 0:
                logger.info(
                    "Observed %s new terminal results during submission cooldown; clearing cooldown and refilling slots.",
                    poll_outcome.new_result_count,
                )
            else:
                return self._defer_pending_timeouts_for_wait(
                    run_id=runtime.service_run_id,
                    updated_at=now,
                    outcome=self._with_queue_metrics(
                        replace(
                            poll_outcome,
                            status="cooldown",
                            cooldown_until=runtime.cooldown_until,
                        ),
                        service_name=runtime.service_name,
                        run_id=runtime.service_run_id,
                    ),
                )

        try:
            dispatch_outcome = self._prepare_and_refill_dispatch_queue(
                runtime=runtime,
                tick_id=tick_id,
                pending_cap=self._effective_pending_cap(
                    runtime=runtime,
                    now=now,
                    allow_observed_limit_probe=poll_outcome.new_result_count > 0,
                ),
                now=now,
            )
        except ConcurrentSimulationLimitExceeded as exc:
            return self._submission_cooldown_outcome(runtime=runtime, now=now, error=exc, logger=logger)
        return self._merge_tick_outcomes(
            poll_outcome=poll_outcome,
            submit_outcome=dispatch_outcome,
        )

    def _poll_pending_batches(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
        batch_ids: list[str],
    ) -> ServiceTickOutcome:
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-poll",
            tick_id=tick_id,
        )
        poll_started = time.perf_counter()
        completed_results: list[SimulationResult] = []
        failed_results: list[SimulationResult] = []
        new_result_count = 0
        active_batch_id = batch_ids[0] if batch_ids else None

        for batch_id in batch_ids:
            refreshed = self.brain_service.poll_batch_once(
                batch_id,
                config=self.config,
                environment=self.environment,
                stuck_job_after_seconds=self.config.service.stuck_job_after_seconds,
            )
            logger.info(
                "Polled batch=%s status=%s pending=%s new_results=%s",
                batch_id,
                refreshed.status,
                refreshed.pending_count,
                len(refreshed.results),
            )
            new_result_count += len(refreshed.results)
            completed_count = sum(1 for result in refreshed.results if result.status == "completed")
            failed_count = sum(1 for result in refreshed.results if result.status in {"failed", "rejected", "timeout"})
            if refreshed.results or refreshed.status == "completed":
                append_progress_event(
                    self.config,
                    self.environment,
                    event="batch_polled",
                    stage="service-poll",
                    status=refreshed.status,
                    tick_id=tick_id,
                    round_index=refreshed.jobs[0].round_index if refreshed.jobs else None,
                    batch_id=batch_id,
                    payload={
                        "pending_count": refreshed.pending_count,
                        "new_result_count": len(refreshed.results),
                        "completed_count": completed_count,
                        "failed_count": failed_count,
                    },
                )
            for result in refreshed.results:
                if result.status == "completed":
                    completed_results.append(result)
                elif result.status in {"failed", "rejected", "timeout"}:
                    failed_results.append(result)
            if refreshed.status == "completed":
                self._learn_from_completed_batch(run_id=runtime.service_run_id, batch_id=batch_id)
            self._sync_service_round(
                run_id=runtime.service_run_id,
                batch_id=batch_id,
                status_override=refreshed.status,
            )

        pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        next_active_batch = None
        if pending_count:
            next_pending = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            next_active_batch = next_pending[0].batch_id if next_pending else active_batch_id
        status = "running" if pending_count else "idle"
        return self._with_queue_metrics(
            ServiceTickOutcome(
                status=status,
                pending_job_count=pending_count,
                new_result_count=new_result_count,
                active_batch_id=next_active_batch,
                completed_count=len(completed_results),
                failed_count=len(failed_results),
                poll_pending_ms=(time.perf_counter() - poll_started) * 1000.0,
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

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

    def _probe_auth_after_batch_failures(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        logger,
    ) -> ServiceTickOutcome | None:
        threshold = self.config.service.max_consecutive_batch_failures_before_auth_check
        recent_all_failed = self.repository.submissions.count_recent_all_failed_batches(
            runtime.service_run_id,
            lookback=threshold + 1,
        )
        if recent_all_failed < threshold:
            return None
        logger.warning(
            "Detected %s consecutive batches with 100%% job failures. Probing auth session before preparing more work.",
            recent_all_failed,
        )
        session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=False)
        if session_state.status == "ready":
            return None
        logger.warning(
            "Auth probe after batch failures detected session status=%s. Triggering re-authentication flow.",
            session_state.status,
        )
        now = datetime.now(UTC).isoformat()
        self.repository.service_runtime.update_state(
            runtime.service_name,
            status="waiting_persona",
            persona_wait_started_at=now,
            updated_at=now,
        )
        pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        return self._with_queue_metrics(
            self._defer_pending_timeouts_for_wait(
                run_id=runtime.service_run_id,
                updated_at=now,
                outcome=ServiceTickOutcome(
                    status="waiting_persona",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    last_error=f"Auth re-check triggered after {recent_all_failed} consecutive all-failed batches.",
                    next_sleep_seconds=self.config.service.persona_retry_interval_seconds,
                ),
            ),
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
        )

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

    @staticmethod
    def _parse_datetime(timestamp: str | None) -> datetime | None:
        if not timestamp:
            return None
        try:
            return datetime.fromisoformat(timestamp)
        except ValueError:
            return None

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

    def _recover_submitting_batches(
        self,
        *,
        service_name: str,
        run_id: str,
        now: str,
        allow_resubmit: bool = True,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        recovered: list[str] = []
        failed: list[str] = []
        resubmitted: list[str] = []
        quarantined: list[str] = []
        policy = self.config.service.ambiguous_submission_policy
        statuses: tuple[str, ...] = ("submitting", "paused_quarantine") if policy in {"fail", "resubmit"} else ("submitting",)
        batches = self.repository.submissions.list_batches_by_status(run_id=run_id, statuses=statuses)
        submissions_by_batch = {
            batch.batch_id: self.repository.submissions.list_submissions(run_id=run_id, batch_id=batch.batch_id)
            for batch in batches
        }
        superseded_batch_ids = self._resubmitted_source_batch_ids(run_id=run_id)
        latest_resubmit_batch_id = None
        if policy == "resubmit" and allow_resubmit:
            latest_resubmit_batch_id = self._latest_resubmittable_ambiguous_batch_id(
                batches=batches,
                submissions_by_batch=submissions_by_batch,
                superseded_batch_ids=superseded_batch_ids,
            )
        for batch in batches:
            if batch.status == "paused_quarantine" and not self._is_ambiguous_batch(batch):
                quarantined.append(batch.batch_id)
                continue
            submissions = submissions_by_batch[batch.batch_id]
            should_recover = bool(
                submissions
                and (
                    policy == "resubmit"
                    or (batch.candidate_count > 0 and len(submissions) >= batch.candidate_count)
                )
            )
            if should_recover:
                recovered_status = "manual_pending" if all(
                    submission.status == "manual_pending" for submission in submissions
                ) else "submitted"
                self.repository.submissions.update_batch_status(
                    batch.batch_id,
                    status=recovered_status,
                    updated_at=now,
                    service_status_reason=None,
                )
                recovered.append(batch.batch_id)
                continue
            if policy == "resubmit" and batch.batch_id in superseded_batch_ids:
                self._mark_ambiguous_batch_failed(
                    batch=batch,
                    submissions=submissions,
                    updated_at=now,
                    reason="ambiguous_submission_superseded",
                )
                failed.append(batch.batch_id)
                continue
            if policy == "resubmit" and not allow_resubmit:
                continue
            if (
                policy == "resubmit"
                and latest_resubmit_batch_id is not None
                and batch.batch_id != latest_resubmit_batch_id
            ):
                self._mark_ambiguous_batch_failed(
                    batch=batch,
                    submissions=submissions,
                    updated_at=now,
                    reason="ambiguous_submission_stale",
                )
                failed.append(batch.batch_id)
                continue
            if policy == "resubmit":
                resubmitted_batch_id = self._resubmit_ambiguous_batch(
                    run_id=run_id,
                    batch=batch,
                    submissions=submissions,
                )
                if resubmitted_batch_id:
                    resubmitted.append(resubmitted_batch_id)
                else:
                    failed.append(batch.batch_id)
                continue
            if policy == "quarantine":
                self.repository.submissions.update_batch_status(
                    batch.batch_id,
                    status="paused_quarantine",
                    updated_at=now,
                    service_status_reason="ambiguous_submission",
                    quarantined_at=now,
                )
                self.repository.service_runtime.update_state(
                    service_name,
                    status="paused_quarantine",
                    active_batch_id=batch.batch_id,
                    updated_at=now,
                )
                quarantined.append(batch.batch_id)
                continue
            self._mark_ambiguous_batch_failed(
                batch=batch,
                submissions=submissions,
                updated_at=now,
                reason="ambiguous_submission_assumed_failed",
            )
            failed.append(batch.batch_id)
        return recovered, failed, resubmitted, quarantined

    @staticmethod
    def _is_ambiguous_batch(batch: SubmissionBatchRecord) -> bool:
        reason = str(batch.service_status_reason or "")
        return batch.status == "submitting" or reason.startswith("ambiguous_submission")

    def _resubmit_ambiguous_batch(
        self,
        *,
        run_id: str,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> str | None:
        candidate_ids = self._candidate_ids_for_batch(batch=batch, submissions=submissions)
        loaded_candidates, skipped_candidates = self._load_candidates_for_batch(
            run_id=run_id,
            batch=batch,
            submissions=submissions,
        )
        active_submissions = self._active_submissions_by_candidate(run_id=run_id, excluding_batch_id=batch.batch_id)
        terminal_results = self._terminal_results_by_candidate(run_id=run_id)
        candidate_failure_reasons = {
            str(item["candidate_id"]): str(item["reason"])
            for item in skipped_candidates
            if item.get("candidate_id") and item.get("reason")
        }
        resubmittable_candidates: list[AlphaCandidate] = []
        for candidate_id in candidate_ids:
            if candidate_id in candidate_failure_reasons:
                continue
            active_submission = active_submissions.get(candidate_id)
            if active_submission is not None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_replay_guard_active_submission",
                        "blocking_batch_id": active_submission.batch_id,
                        "blocking_job_id": active_submission.job_id,
                        "blocking_status": active_submission.status,
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_replay_guard_active_submission"
                continue
            terminal_result = terminal_results.get(candidate_id)
            if terminal_result is not None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_replay_guard_terminal_result",
                        "blocking_batch_id": terminal_result.batch_id,
                        "blocking_job_id": terminal_result.job_id,
                        "blocking_status": terminal_result.status,
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_replay_guard_terminal_result"
                continue
            candidate = loaded_candidates.get(candidate_id)
            if candidate is None:
                skipped_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "reason": "ambiguous_submission_missing_candidate",
                    }
                )
                candidate_failure_reasons[candidate_id] = "ambiguous_submission_missing_candidate"
                continue
            resubmittable_candidates.append(candidate)
        recovery_notes = self._ambiguous_recovery_notes(
            source_candidate_ids=candidate_ids,
            skipped_candidates=skipped_candidates,
        )
        updated_at = datetime.now(UTC).isoformat()
        if not resubmittable_candidates:
            self._mark_ambiguous_batch_failed(
                batch=batch,
                submissions=submissions,
                updated_at=updated_at,
                reason="ambiguous_submission_no_resubmittable_candidates",
                note_overrides=recovery_notes,
                candidate_failure_reasons=candidate_failure_reasons,
            )
            return None
        sim_config_override = self._decode_json_object(batch.sim_config_snapshot)
        resubmitted_batch = self.brain_service.submit_candidates(
            resubmittable_candidates,
            config=self.config,
            environment=self.environment,
            round_index=batch.round_index,
            batch_size=len(resubmittable_candidates),
            sim_config_override=sim_config_override,
            note_overrides={
                "resubmitted_from_batch_id": batch.batch_id,
                **recovery_notes,
            },
        )
        self._mark_ambiguous_batch_failed(
            batch=batch,
            submissions=submissions,
            updated_at=updated_at,
            reason=f"ambiguous_submission_resubmitted:{resubmitted_batch.batch_id}",
            note_overrides={
                **recovery_notes,
                "resubmitted_batch_id": resubmitted_batch.batch_id,
            },
            candidate_failure_reasons=candidate_failure_reasons,
        )
        return resubmitted_batch.batch_id

    def _mark_ambiguous_batch_failed(
        self,
        *,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
        updated_at: str,
        reason: str,
        note_overrides: dict[str, object] | None = None,
        candidate_failure_reasons: dict[str, str] | None = None,
    ) -> None:
        for submission in submissions:
            if submission.status not in {"submitted", "running", "manual_pending"}:
                continue
            submission_reason = (
                candidate_failure_reasons.get(submission.candidate_id, reason)
                if candidate_failure_reasons is not None
                else reason
            )
            self.repository.submissions.update_submission_runtime(
                submission.job_id,
                status="failed",
                updated_at=updated_at,
                completed_at=updated_at,
                error_message=submission_reason,
                last_polled_at=updated_at,
                next_poll_after=None,
                stuck_since=None,
                service_failure_reason=submission_reason,
            )
        notes_json = self._merged_batch_notes(batch=batch, note_overrides=note_overrides)
        update_kwargs: dict[str, object] = {
            "status": "failed",
            "updated_at": updated_at,
            "service_status_reason": reason,
            "quarantined_at": None,
        }
        if notes_json is not None:
            update_kwargs["notes_json"] = notes_json
        self.repository.submissions.update_batch_status(
            batch.batch_id,
            **update_kwargs,
        )

    def _load_candidates_for_batch(
        self,
        *,
        run_id: str,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> tuple[dict[str, AlphaCandidate], list[dict[str, str]]]:
        candidate_ids = self._candidate_ids_for_batch(batch=batch, submissions=submissions)
        if not candidate_ids:
            return {}, []
        payloads_by_candidate = self._batch_payloads_by_candidate(batch)
        submissions_by_candidate = {submission.candidate_id: submission for submission in submissions if submission.candidate_id}
        parent_refs_map = self.repository.get_parent_refs(run_id)
        records_by_id = {
            record.alpha_id: alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id))
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in set(candidate_ids)
        }
        candidates: dict[str, AlphaCandidate] = {}
        skipped: list[dict[str, str]] = []
        for candidate_id in candidate_ids:
            candidate = records_by_id.get(candidate_id)
            if candidate is None:
                payload = payloads_by_candidate.get(candidate_id, {})
                submission = submissions_by_candidate.get(candidate_id)
                expression = str(payload.get("expression") or (submission.expression if submission else "")).strip()
                if not expression:
                    skipped.append(
                        {
                            "candidate_id": candidate_id,
                            "reason": "ambiguous_submission_missing_candidate",
                        }
                    )
                    continue
                generation_metadata = payload.get("generation_metadata")
                candidate = AlphaCandidate(
                    alpha_id=candidate_id,
                    expression=expression,
                    normalized_expression=expression,
                    generation_mode=str(payload.get("generation_mode") or "recovered"),
                    parent_ids=(),
                    complexity=int(payload.get("complexity") or 0),
                    created_at=batch.created_at,
                    template_name=str(payload.get("template_name") or ""),
                    fields_used=tuple(str(item) for item in (payload.get("fields_used") or ()) if str(item)),
                    operators_used=tuple(str(item) for item in (payload.get("operators_used") or ()) if str(item)),
                    depth=int(payload.get("depth") or 0),
                    generation_metadata=generation_metadata if isinstance(generation_metadata, dict) else {},
                )
            candidates[candidate_id] = candidate
        return candidates, skipped

    def _candidate_ids_for_batch(
        self,
        *,
        batch: SubmissionBatchRecord,
        submissions: list[SubmissionRecord],
    ) -> list[str]:
        candidate_ids: list[str] = []
        for payload in self._decode_json_object(batch.sim_config_snapshot).get("candidate_payloads", []):
            if isinstance(payload, dict) and payload.get("candidate_id"):
                candidate_ids.append(str(payload["candidate_id"]))
        if not candidate_ids:
            for candidate_id in self._decode_json_object(batch.notes_json).get("candidate_ids", []):
                if candidate_id:
                    candidate_ids.append(str(candidate_id))
        if not candidate_ids:
            candidate_ids.extend(submission.candidate_id for submission in submissions if submission.candidate_id)
        return list(dict.fromkeys(candidate_ids))

    def _batch_payloads_by_candidate(self, batch: SubmissionBatchRecord) -> dict[str, dict[str, object]]:
        payloads = self._decode_json_object(batch.sim_config_snapshot).get("candidate_payloads", [])
        if not isinstance(payloads, list):
            return {}
        return {
            str(item.get("candidate_id")): dict(item)
            for item in payloads
            if isinstance(item, dict) and item.get("candidate_id")
        }

    def _active_submissions_by_candidate(
        self,
        *,
        run_id: str,
        excluding_batch_id: str,
    ) -> dict[str, SubmissionRecord]:
        active: dict[str, SubmissionRecord] = {}
        for submission in self.repository.submissions.list_pending_submissions(run_id):
            if not submission.candidate_id or submission.batch_id == excluding_batch_id:
                continue
            active.setdefault(submission.candidate_id, submission)
        return active

    def _terminal_results_by_candidate(self, *, run_id: str) -> dict[str, BrainResultRecord]:
        terminal_statuses = {"completed", "failed", "rejected", "timeout"}
        return {
            result.candidate_id: result
            for result in self.repository.brain_results.list_latest_results_by_candidate(run_id)
            if result.candidate_id and result.status in terminal_statuses
        }

    @staticmethod
    def _ambiguous_recovery_notes(
        *,
        source_candidate_ids: list[str],
        skipped_candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        notes: dict[str, object] = {
            "recovery_source_candidate_ids": list(source_candidate_ids),
        }
        if skipped_candidates:
            notes["recovery_skipped_candidates"] = [dict(item) for item in skipped_candidates]
        return notes

    def _merged_batch_notes(
        self,
        *,
        batch: SubmissionBatchRecord,
        note_overrides: dict[str, object] | None,
    ) -> str | None:
        if note_overrides is None:
            return None
        notes = self._decode_json_object(batch.notes_json)
        notes.update(note_overrides)
        return json.dumps(notes, sort_keys=True)

    def _resubmitted_source_batch_ids(self, *, run_id: str) -> set[str]:
        source_batch_ids: set[str] = set()
        for batch in self.repository.submissions.list_batches(run_id):
            resubmitted_from = str(
                self._decode_json_object(batch.notes_json).get("resubmitted_from_batch_id") or ""
            ).strip()
            if resubmitted_from:
                source_batch_ids.add(resubmitted_from)
        return source_batch_ids

    def _latest_resubmittable_ambiguous_batch_id(
        self,
        *,
        batches: list[SubmissionBatchRecord],
        submissions_by_batch: dict[str, list[SubmissionRecord]],
        superseded_batch_ids: set[str],
    ) -> str | None:
        candidates: list[SubmissionBatchRecord] = []
        for batch in batches:
            if not self._is_ambiguous_batch(batch):
                continue
            if batch.batch_id in superseded_batch_ids:
                continue
            submissions = submissions_by_batch.get(batch.batch_id, [])
            if batch.candidate_count > 0 and len(submissions) >= batch.candidate_count:
                continue
            candidates.append(batch)
        if not candidates:
            return None
        latest = max(candidates, key=self._batch_recency_key)
        return latest.batch_id

    @staticmethod
    def _batch_recency_key(batch: SubmissionBatchRecord) -> tuple[datetime, datetime, int, str]:
        baseline = datetime.min.replace(tzinfo=UTC)

        def _parse(timestamp: str | None) -> datetime:
            if not timestamp:
                return baseline
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                return baseline

        return (_parse(batch.created_at), _parse(batch.updated_at), int(batch.round_index), batch.batch_id)

    @staticmethod
    def _decode_json_object(payload: str | None) -> dict[str, object]:
        if not payload:
            return {}
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def _learn_from_completed_batch(self, *, run_id: str, batch_id: str) -> None:
        results = [
            record
            for record in self.repository.brain_results.list_results(run_id=run_id)
            if record.batch_id == batch_id
        ]
        if not results:
            return
        regime_key = self._resolve_regime_key(run_id)
        run = self.repository.get_run(run_id)
        region = str(run.region or "") if run else ""
        global_regime_key = str(run.global_regime_key or "") if run else ""
        market_regime_key = str(run.market_regime_key or "") if run else ""
        regime_label = str(run.regime_label or "unknown") if run else "unknown"
        regime_confidence = float(run.regime_confidence or 0.0) if run else 0.0
        candidates_by_id = self._load_candidates_by_ids(
            run_id=run_id,
            candidate_ids={result.candidate_id for result in results},
        )
        snapshot = self.repository.alpha_history.load_snapshot(
            regime_key=regime_key,
            region=region,
            global_regime_key=global_regime_key,
            parent_pool_size=max(self.config.adaptive_generation.parent_pool_size, self.config.loop.mutate_top_k * 2),
            region_learning_config=self.config.adaptive_generation.region_learning,
            pattern_decay=self.config.adaptive_generation.pattern_decay,
            prior_weight=self.config.adaptive_generation.critic_thresholds.score_prior_weight,
        )
        case_snapshot = self.repository.alpha_history.load_case_snapshot(
            regime_key,
            region=region,
            global_regime_key=global_regime_key,
            region_learning_config=self.config.adaptive_generation.region_learning,
        )
        selected_parent_results, _, _ = self.selection_service.select_results_for_mutation_with_details(
            [self._simulation_result_from_record(result) for result in results],
            candidates_by_id=candidates_by_id,
            top_k=self.config.loop.mutate_top_k,
            diversity_config=self.config.adaptive_generation.diversity,
            case_snapshot=case_snapshot,
            run_id=run_id,
            round_index=results[0].round_index if results else 0,
        )
        selected_parent_ids = {result.candidate_id for result in selected_parent_results}
        self.learning_service.persist_results(
            config=self.config,
            regime_key=regime_key,
            region=region,
            global_regime_key=global_regime_key,
            market_regime_key=market_regime_key,
            effective_regime_key=regime_key,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
            snapshot=snapshot,
            candidates_by_id=candidates_by_id,
            results=[self._simulation_result_from_record(result) for result in results],
            selected_parent_ids=selected_parent_ids,
        )
        self._sync_service_round(
            run_id=run_id,
            batch_id=batch_id,
            selected_for_mutation_count=len(selected_parent_ids),
        )

    def _reconcile_completed_batches(self, *, run_id: str) -> int:
        reconciled = 0
        completed_batches = self.repository.submissions.list_batches_by_status(run_id=run_id, statuses=("completed",))
        for batch in completed_batches:
            if not self._batch_learning_needed(run_id=run_id, batch_id=batch.batch_id):
                continue
            self._learn_from_completed_batch(run_id=run_id, batch_id=batch.batch_id)
            reconciled += 1
        return reconciled

    def _select_mutation_parent_ids(self, *, run_id: str) -> set[str]:
        latest_results = self.repository.brain_results.list_latest_results_by_candidate(run_id)
        if not latest_results:
            return set()
        regime_key = self._resolve_regime_key(run_id)
        run = self.repository.get_run(run_id)
        region = str(run.region or "") if run else ""
        global_regime_key = str(run.global_regime_key or "") if run else ""
        case_snapshot = self.repository.alpha_history.load_case_snapshot(
            regime_key,
            region=region,
            global_regime_key=global_regime_key,
            region_learning_config=self.config.adaptive_generation.region_learning,
        )
        candidates_by_id = self._load_candidates_by_ids(
            run_id=run_id,
            candidate_ids={result.candidate_id for result in latest_results},
        )
        selected = self.selection_service.select_results_for_mutation(
            [self._simulation_result_from_record(result) for result in latest_results],
            candidates_by_id=candidates_by_id,
            top_k=self.config.loop.mutate_top_k,
            diversity_config=self.config.adaptive_generation.diversity,
            case_snapshot=case_snapshot,
        )
        return {result.candidate_id for result in selected}

    def _load_candidates_by_ids(self, *, run_id: str, candidate_ids: set[str]) -> dict[str, AlphaCandidate]:
        if not candidate_ids:
            return {}
        parent_refs_map = self.repository.get_parent_refs(run_id)
        return {
            record.alpha_id: alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id))
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in candidate_ids
        }

    def _resolve_regime_key(self, run_id: str) -> str:
        run = self.repository.get_run(run_id)
        if run and (run.effective_regime_key or run.regime_key):
            return str(run.effective_regime_key or run.regime_key)
        research_context = load_research_context(
            self.config,
            self.environment,
            stage="service-learning-data",
        )
        return research_context.effective_regime_key or research_context.regime_key

    def _simulation_result_from_record(self, record: BrainResultRecord) -> SimulationResult:
        return SimulationResult(
            expression=record.expression,
            job_id=record.job_id,
            status=record.status,
            region=record.region,
            universe=record.universe,
            delay=record.delay,
            neutralization=record.neutralization,
            decay=record.decay,
            metrics={
                "sharpe": record.sharpe,
                "fitness": record.fitness,
                "turnover": record.turnover,
                "drawdown": record.drawdown,
                "returns": record.returns,
                "margin": record.margin,
            },
            submission_eligible=record.submission_eligible,
            rejection_reason=record.rejection_reason,
            raw_result={},
            simulated_at=record.simulated_at,
            candidate_id=record.candidate_id,
            batch_id=record.batch_id,
            run_id=record.run_id,
            round_index=record.round_index,
            backend=self.config.brain.backend,
        )

    def _batch_learning_needed(self, *, run_id: str, batch_id: str) -> bool:
        candidate_rows = self.repository.connection.execute(
            """
            SELECT DISTINCT candidate_id
            FROM brain_results
            WHERE run_id = ? AND batch_id = ?
            """,
            (run_id, batch_id),
        ).fetchall()
        candidate_ids = [str(row["candidate_id"]) for row in candidate_rows if row["candidate_id"]]
        if not candidate_ids:
            return False
        placeholders = ", ".join("?" for _ in candidate_ids)
        learned_rows = self.repository.connection.execute(
            f"""
            SELECT COUNT(DISTINCT alpha_id) AS total
            FROM alpha_history
            WHERE run_id = ? AND metric_source = 'external_brain' AND alpha_id IN ({placeholders})
            """,
            (run_id, *candidate_ids),
        ).fetchone()
        learned_total = int(learned_rows["total"] or 0)
        return learned_total < len(candidate_ids)

    def _sync_service_round(
        self,
        *,
        run_id: str,
        batch_id: str | None = None,
        round_index_override: int | None = None,
        generated_count: int | None = None,
        validated_count: int | None = None,
        submitted_count: int | None = None,
        selected_for_mutation_count: int | None = None,
        mutated_children_count: int | None = None,
        status_override: str | None = None,
        note_overrides: dict[str, object] | None = None,
    ) -> None:
        batch = self.repository.submissions.get_batch(batch_id) if batch_id is not None else None
        if batch is None and round_index_override is None:
            return
        round_index = int(round_index_override if round_index_override is not None else batch.round_index)
        existing = self.repository.brain_results.get_closed_loop_round(run_id, round_index)
        round_batches = [item for item in self.repository.submissions.list_batches(run_id) if int(item.round_index) == round_index]
        submissions = self.repository.submissions.list_submissions(run_id=run_id, round_index=round_index)
        terminal_statuses = {"completed", "failed", "rejected", "timeout"}
        terminal_submission_count = sum(1 for submission in submissions if submission.status in terminal_statuses)
        timestamp = datetime.now(UTC).isoformat()
        latest_batch = batch or (max(round_batches, key=self._batch_recency_key) if round_batches else None)

        summary_payload: dict[str, object] = {}
        if existing is not None and existing.summary_json:
            try:
                decoded = json.loads(existing.summary_json)
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict):
                summary_payload.update(decoded)
        summary_payload.update(
            {
                "source": "service",
                "round_index": round_index,
                "batch_id": latest_batch.batch_id if latest_batch is not None else summary_payload.get("batch_id"),
                "backend": latest_batch.backend if latest_batch is not None else self.config.brain.backend,
                "export_path": latest_batch.export_path if latest_batch is not None else summary_payload.get("export_path"),
                "service_status_reason": (
                    latest_batch.service_status_reason if latest_batch is not None else summary_payload.get("service_status_reason")
                ),
                "candidate_count": sum(int(item.candidate_count or 0) for item in round_batches),
                "batch_count": len(round_batches),
                "terminal_submission_count": terminal_submission_count,
                "submission_status_counts": dict(
                    Counter(submission.status for submission in submissions if submission.status)
                ),
                "queue_status_counts": self._queue_counts(
                    service_name=self.config.service.lock_name,
                    run_id=run_id,
                    source_round_index=round_index,
                ),
            }
        )
        if note_overrides:
            summary_payload.update(note_overrides)

        generated_total = generated_count
        if generated_total is None:
            generated_total = existing.generated_count if existing is not None else int(summary_payload.get("candidate_count") or 0)
        validated_total = validated_count
        if validated_total is None:
            validated_total = existing.validated_count if existing is not None else generated_total
        submitted_total = submitted_count
        if submitted_total is None:
            submitted_total = len(submissions) or (existing.submitted_count if existing is not None else 0)
        selected_total = selected_for_mutation_count
        if selected_total is None:
            selected_total = existing.selected_for_mutation_count if existing is not None else self._round_selected_parent_count(
                run_id=run_id,
                candidate_ids=[submission.candidate_id for submission in submissions if submission.candidate_id],
            )
        mutated_total = mutated_children_count
        if mutated_total is None:
            mutated_total = existing.mutated_children_count if existing is not None else 0

        self.repository.brain_results.upsert_closed_loop_round(
            ClosedLoopRoundRecord(
                run_id=run_id,
                round_index=round_index,
                status=self._derive_service_round_status(
                    round_batches=round_batches,
                    fallback=status_override or (existing.status if existing is not None else "queued"),
                ),
                generated_count=int(generated_total or 0),
                validated_count=int(validated_total or 0),
                submitted_count=int(submitted_total or 0),
                completed_count=int(terminal_submission_count),
                selected_for_mutation_count=int(selected_total or 0),
                mutated_children_count=int(mutated_total or 0),
                summary_json=json.dumps(summary_payload, sort_keys=True),
                created_at=existing.created_at if existing is not None else timestamp,
                updated_at=timestamp,
            )
        )

    def _derive_service_round_status(
        self,
        *,
        round_batches: list[SubmissionBatchRecord],
        fallback: str,
    ) -> str:
        if not round_batches:
            return fallback
        statuses = [str(batch.status or "") for batch in round_batches if batch.status]
        if any(status in {"submitting", "submitted", "running"} for status in statuses):
            return "running"
        if statuses and all(status == "manual_pending" for status in statuses):
            return "manual_pending"
        if any(status == "paused_quarantine" for status in statuses):
            return "paused_quarantine"
        if statuses and all(status == "completed" for status in statuses):
            return "completed"
        if any(status == "failed" for status in statuses):
            return "failed"
        latest_batch = max(round_batches, key=self._batch_recency_key)
        return str(latest_batch.status or fallback)

    def _round_selected_parent_count(self, *, run_id: str, candidate_ids: list[str]) -> int:
        normalized_candidate_ids = tuple(dict.fromkeys(candidate_id for candidate_id in candidate_ids if candidate_id))
        if not normalized_candidate_ids:
            return 0
        placeholders = ", ".join("?" for _ in normalized_candidate_ids)
        row = self.repository.connection.execute(
            f"""
            SELECT COUNT(DISTINCT alpha_id) AS total
            FROM alpha_history
            WHERE run_id = ?
              AND metric_source = 'external_brain'
              AND selected = 1
              AND alpha_id IN ({placeholders})
            """,
            (run_id, *normalized_candidate_ids),
        ).fetchone()
        return int(row["total"] or 0) if row is not None else 0

    @staticmethod
    def _should_probe_session_during_cooldown(runtime: ServiceRuntimeRecord) -> bool:
        if runtime.status in {
            "waiting_persona",
            "waiting_persona_confirmation",
            "auth_throttled",
            "auth_unavailable",
        }:
            return True
        if runtime.persona_url or runtime.persona_wait_started_at or runtime.persona_confirmation_nonce:
            return True
        last_error = str(runtime.last_error or "").upper()
        return any(
            token in last_error
            for token in (
                "BRAIN AUTHENTICATION FAILED",
                "BRAIN AUTH TRANSPORT ERROR",
                "BIOMETRICS_THROTTLED",
                "PERSONA",
            )
        )

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

    def _update_persona_confirmation_state(
        self,
        *,
        service_name: str,
        now: str,
        nonce: str | None,
        last_prompt_at: str | None,
        granted_at: str | None,
        last_update_id: int | None,
    ) -> None:
        self.repository.service_runtime.update_state(
            service_name,
            persona_confirmation_nonce=nonce,
            persona_confirmation_last_prompt_at=last_prompt_at,
            persona_confirmation_granted_at=granted_at,
            persona_confirmation_last_update_id=last_update_id,
            updated_at=now,
        )
