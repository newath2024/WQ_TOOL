from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from domain.exceptions import ConcurrentSimulationLimitExceeded
from core.config import AppConfig
from core.logging import get_logger
from services.brain_batch_service import BrainBatchService
from services.brain_learning_service import BrainLearningService
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.models import CommandEnvironment, ServiceTickOutcome
from services.notification_manager import NotificationManager
from services.session_manager import SessionManager
from storage.models import (
    ServiceRuntimeRecord,
)
from storage.repository import SQLiteRepository
from services.service_runtime.auth_flow import AuthFlow
from services.service_runtime.batch_poller import BatchPoller
from services.service_runtime.batch_recovery import BatchRecovery
from services.service_runtime.batch_submitter import BatchSubmitter
from services.service_runtime.cooldown_policy import CooldownPolicy
from services.service_runtime.learning_coordinator import LearningCoordinator
from services.service_runtime.queue_manager import QueueManager
from services.service_runtime.telemetry import Telemetry


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
        self.auth_flow = AuthFlow(self)
        self.cooldown_policy = CooldownPolicy(self)
        self.queue_manager = QueueManager(self)
        self.batch_submitter = BatchSubmitter(self)
        self.batch_poller = BatchPoller(self)
        self.batch_recovery = BatchRecovery(self)
        self.learning_coordinator = LearningCoordinator(self)
        self.telemetry = Telemetry(self)

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
        runtime, submission_cooldown_active, auth_outcome = self.auth_flow.prepare_session(
            runtime=runtime,
            now=now,
            logger=logger,
        )
        if auth_outcome is not None:
            return auth_outcome

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

    def _poll_pending_batches(self, *args, **kwargs):
        return self.batch_poller._poll_pending_batches(*args, **kwargs)

    def _prepare_and_refill_dispatch_queue(self, *args, **kwargs):
        return self.queue_manager._prepare_and_refill_dispatch_queue(*args, **kwargs)

    def _submit_new_batch(self, *args, **kwargs):
        return self.batch_submitter._submit_new_batch(*args, **kwargs)

    def _prepare_dispatch_queue_if_needed(self, *args, **kwargs):
        return self.queue_manager._prepare_dispatch_queue_if_needed(*args, **kwargs)

    def _dispatch_queued_candidates(self, *args, **kwargs):
        return self.batch_submitter._dispatch_queued_candidates(*args, **kwargs)

    def _submission_cooldown_outcome(self, *args, **kwargs):
        return self.cooldown_policy._submission_cooldown_outcome(*args, **kwargs)

    def _merge_tick_outcomes(self, *args, **kwargs):
        return self.telemetry._merge_tick_outcomes(*args, **kwargs)

    def _defer_pending_timeouts_for_wait(self, *args, **kwargs):
        return self.cooldown_policy._defer_pending_timeouts_for_wait(*args, **kwargs)

    def _with_queue_metrics(self, *args, **kwargs):
        return self.telemetry._with_queue_metrics(*args, **kwargs)

    def _queue_counts(self, *args, **kwargs):
        return self.telemetry._queue_counts(*args, **kwargs)

    def _queue_depth(self, *args, **kwargs):
        return self.telemetry._queue_depth(*args, **kwargs)

    def _recover_dispatch_queue_items(self, *args, **kwargs):
        return self.queue_manager._recover_dispatch_queue_items(*args, **kwargs)

    def _queue_item_id(self, *args, **kwargs):
        return self.queue_manager._queue_item_id(*args, **kwargs)

    def _next_service_round_index(self, *args, **kwargs):
        return self.queue_manager._next_service_round_index(*args, **kwargs)

    def _probe_auth_after_batch_failures(self, *args, **kwargs):
        return self.auth_flow._probe_auth_after_batch_failures(*args, **kwargs)

    def _is_dispatch_blocked_status(self, *args, **kwargs):
        return self.cooldown_policy._is_dispatch_blocked_status(*args, **kwargs)

    def _effective_pending_cap(self, *args, **kwargs):
        return self.cooldown_policy._effective_pending_cap(*args, **kwargs)

    def _observed_limit_state(self, *args, **kwargs):
        return self.cooldown_policy._observed_limit_state(*args, **kwargs)

    def _observed_limit_probe_due(self, *args, **kwargs):
        return self.cooldown_policy._observed_limit_probe_due(*args, **kwargs)

    def _mark_observed_limit_probe_if_needed(self, *args, **kwargs):
        return self.cooldown_policy._mark_observed_limit_probe_if_needed(*args, **kwargs)

    def _clear_observed_limit(self, *args, **kwargs):
        return self.cooldown_policy._clear_observed_limit(*args, **kwargs)

    def _parse_datetime(self, *args, **kwargs):
        return self.telemetry._parse_datetime(*args, **kwargs)

    def _derive_observed_limit_telemetry(self, *args, **kwargs):
        return self.cooldown_policy._derive_observed_limit_telemetry(*args, **kwargs)

    def _latest_interrupted_limit_batch(self, *args, **kwargs):
        return self.cooldown_policy._latest_interrupted_limit_batch(*args, **kwargs)

    def _partial_submitted_count(self, *args, **kwargs):
        return self.cooldown_policy._partial_submitted_count(*args, **kwargs)

    def _persist_limit_hit_telemetry(self, *args, **kwargs):
        return self.cooldown_policy._persist_limit_hit_telemetry(*args, **kwargs)

    def _recover_submitting_batches(self, *args, **kwargs):
        return self.batch_recovery._recover_submitting_batches(*args, **kwargs)

    def _is_ambiguous_batch(self, *args, **kwargs):
        return self.batch_recovery._is_ambiguous_batch(*args, **kwargs)

    def _resubmit_ambiguous_batch(self, *args, **kwargs):
        return self.batch_recovery._resubmit_ambiguous_batch(*args, **kwargs)

    def _mark_ambiguous_batch_failed(self, *args, **kwargs):
        return self.batch_recovery._mark_ambiguous_batch_failed(*args, **kwargs)

    def _load_candidates_for_batch(self, *args, **kwargs):
        return self.batch_recovery._load_candidates_for_batch(*args, **kwargs)

    def _candidate_ids_for_batch(self, *args, **kwargs):
        return self.batch_recovery._candidate_ids_for_batch(*args, **kwargs)

    def _batch_payloads_by_candidate(self, *args, **kwargs):
        return self.batch_recovery._batch_payloads_by_candidate(*args, **kwargs)

    def _active_submissions_by_candidate(self, *args, **kwargs):
        return self.batch_recovery._active_submissions_by_candidate(*args, **kwargs)

    def _terminal_results_by_candidate(self, *args, **kwargs):
        return self.batch_recovery._terminal_results_by_candidate(*args, **kwargs)

    def _ambiguous_recovery_notes(self, *args, **kwargs):
        return self.batch_recovery._ambiguous_recovery_notes(*args, **kwargs)

    def _merged_batch_notes(self, *args, **kwargs):
        return self.batch_recovery._merged_batch_notes(*args, **kwargs)

    def _resubmitted_source_batch_ids(self, *args, **kwargs):
        return self.batch_recovery._resubmitted_source_batch_ids(*args, **kwargs)

    def _latest_resubmittable_ambiguous_batch_id(self, *args, **kwargs):
        return self.batch_recovery._latest_resubmittable_ambiguous_batch_id(*args, **kwargs)

    def _batch_recency_key(self, *args, **kwargs):
        return self.batch_recovery._batch_recency_key(*args, **kwargs)

    def _decode_json_object(self, *args, **kwargs):
        return self.telemetry._decode_json_object(*args, **kwargs)

    def _learn_from_completed_batch(self, *args, **kwargs):
        return self.learning_coordinator._learn_from_completed_batch(*args, **kwargs)

    def _reconcile_completed_batches(self, *args, **kwargs):
        return self.learning_coordinator._reconcile_completed_batches(*args, **kwargs)

    def _select_mutation_parent_ids(self, *args, **kwargs):
        return self.learning_coordinator._select_mutation_parent_ids(*args, **kwargs)

    def _load_candidates_by_ids(self, *args, **kwargs):
        return self.learning_coordinator._load_candidates_by_ids(*args, **kwargs)

    def _resolve_regime_key(self, *args, **kwargs):
        return self.learning_coordinator._resolve_regime_key(*args, **kwargs)

    def _simulation_result_from_record(self, *args, **kwargs):
        return self.learning_coordinator._simulation_result_from_record(*args, **kwargs)

    def _batch_learning_needed(self, *args, **kwargs):
        return self.learning_coordinator._batch_learning_needed(*args, **kwargs)

    def _is_neutral_operational_result_row(self, *args, **kwargs):
        return self.learning_coordinator._is_neutral_operational_result_row(*args, **kwargs)

    def _is_neutral_operational_result_record(self, *args, **kwargs):
        return self.learning_coordinator._is_neutral_operational_result_record(*args, **kwargs)

    def _sync_service_round(self, *args, **kwargs):
        return self.learning_coordinator._sync_service_round(*args, **kwargs)

    def _derive_service_round_status(self, *args, **kwargs):
        return self.learning_coordinator._derive_service_round_status(*args, **kwargs)

    def _round_selected_parent_count(self, *args, **kwargs):
        return self.learning_coordinator._round_selected_parent_count(*args, **kwargs)

    def _should_probe_session_during_cooldown(self, *args, **kwargs):
        return self.auth_flow._should_probe_session_during_cooldown(*args, **kwargs)

    def _recent_job_failure_rate(self, *args, **kwargs):
        return self.telemetry._recent_job_failure_rate(*args, **kwargs)

    def _recent_live_timeout_rate(self, *args, **kwargs):
        return self.telemetry._recent_live_timeout_rate(*args, **kwargs)

    def _update_persona_confirmation_state(self, *args, **kwargs):
        return self.auth_flow._update_persona_confirmation_state(*args, **kwargs)

