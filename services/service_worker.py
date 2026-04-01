from __future__ import annotations

import json
from datetime import UTC, datetime

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
from storage.models import BrainResultRecord, ServiceRuntimeRecord
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
        if runtime.cooldown_until and runtime.cooldown_until > now and not should_probe_auth:
            logger.info("Service is cooling down until %s", runtime.cooldown_until)
            return ServiceTickOutcome(
                status="cooldown",
                pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                active_batch_id=runtime.active_batch_id,
                cooldown_until=runtime.cooldown_until,
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
                return ServiceTickOutcome(
                    status="waiting_persona_confirmation",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    persona_url=None,
                    last_error=confirmation.detail,
                    next_sleep_seconds=self.config.service.persona_confirmation_poll_interval_seconds,
                )
            session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=True)
        if session_state.status == "waiting_persona":
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
            return ServiceTickOutcome(
                status="waiting_persona",
                pending_job_count=pending_jobs,
                active_batch_id=runtime.active_batch_id,
                persona_url=session_state.persona_url,
                last_error=session_state.detail,
                next_sleep_seconds=(
                    session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
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
            return ServiceTickOutcome(
                status="auth_throttled",
                pending_job_count=pending_jobs,
                active_batch_id=runtime.active_batch_id,
                persona_url=None,
                last_error=session_state.detail,
                next_sleep_seconds=(
                    session_state.retry_after_seconds or self.config.service.persona_retry_interval_seconds
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
            return ServiceTickOutcome(
                status="auth_unavailable",
                pending_job_count=pending_jobs,
                active_batch_id=runtime.active_batch_id,
                persona_url=None,
                last_error=session_state.detail,
                next_sleep_seconds=self.config.service.tick_interval_seconds,
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

        paused_batches = self.repository.submissions.list_batches_by_status(
            run_id=runtime.service_run_id,
            statuses=("paused_quarantine",),
        )
        if paused_batches:
            active_batch_id = paused_batches[0].batch_id
            logger.error("Service remains paused because batch=%s is quarantined.", active_batch_id)
            return ServiceTickOutcome(
                status="paused_quarantine",
                pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                active_batch_id=active_batch_id,
            )

        recovered_batch_ids, quarantined_batch_ids = self._recover_submitting_batches(
            service_name=runtime.service_name,
            run_id=runtime.service_run_id,
            now=now,
        )
        if recovered_batch_ids:
            logger.info("Recovered %s interrupted submitting batches", len(recovered_batch_ids))
        if quarantined_batch_ids:
            active_batch_id = quarantined_batch_ids[0]
            logger.error("Paused service after quarantining ambiguous batch %s", active_batch_id)
            return ServiceTickOutcome(
                status="paused_quarantine",
                pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                active_batch_id=active_batch_id,
                quarantined_count=len(quarantined_batch_ids),
            )

        pending_batches = self.repository.submissions.list_batches_by_status(
            run_id=runtime.service_run_id,
            statuses=("submitted", "running"),
        )
        if pending_batches:
            return self._poll_pending_batches(
                runtime=runtime,
                tick_id=tick_id,
                batch_ids=[batch.batch_id for batch in pending_batches],
            )

        reconciled_batches = self._reconcile_completed_batches(run_id=runtime.service_run_id)
        if reconciled_batches:
            logger.info("Reconciled learning for %s completed batches before new submission.", reconciled_batches)

        if stop_requested:
            logger.info("Shutdown requested; skipping new batch submission.")
            return ServiceTickOutcome(status="idle", pending_job_count=0)

        return self._submit_new_batch(runtime=runtime, tick_id=tick_id)

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

        pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        next_active_batch = None
        if pending_count:
            next_pending = self.repository.submissions.list_pending_submissions(runtime.service_run_id)
            next_active_batch = next_pending[0].batch_id if next_pending else active_batch_id
        status = "running" if pending_count else "idle"
        return ServiceTickOutcome(
            status=status,
            pending_job_count=pending_count,
            new_result_count=new_result_count,
            active_batch_id=next_active_batch,
            completed_count=len(completed_results),
            failed_count=len(failed_results),
        )

    def _submit_new_batch(
        self,
        *,
        runtime: ServiceRuntimeRecord,
        tick_id: int,
    ) -> ServiceTickOutcome:
        logger = get_logger(
            __name__,
            run_id=runtime.service_run_id,
            stage="service-submit",
            tick_id=tick_id,
        )
        mutation_parent_ids = self._select_mutation_parent_ids(run_id=runtime.service_run_id)
        # --- Pre-submission auth probe on consecutive batch failures ---
        threshold = self.config.service.max_consecutive_batch_failures_before_auth_check
        recent_all_failed = self.repository.submissions.count_recent_all_failed_batches(
            runtime.service_run_id, lookback=threshold + 1,
        )
        if recent_all_failed >= threshold:
            logger.warning(
                "Detected %s consecutive batches with 100%% job failures. "
                "Probing auth session before submitting another batch.",
                recent_all_failed,
            )
            session_state = self.session_manager.ensure_session(runtime=runtime, allow_new_login=False)
            if session_state.status != "ready":
                logger.warning(
                    "Auth probe after batch failures detected session status=%s. "
                    "Triggering re-authentication flow.",
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
                return ServiceTickOutcome(
                    status="waiting_persona",
                    pending_job_count=pending_jobs,
                    active_batch_id=runtime.active_batch_id,
                    last_error=f"Auth re-check triggered after {recent_all_failed} consecutive all-failed batches.",
                    next_sleep_seconds=self.config.service.persona_retry_interval_seconds,
                )

        batch_result = self.batch_service.prepare_service_batch(
            config=self.config,
            environment=self.environment,
            count=self.config.loop.generation_batch_size,
            mutation_parent_ids=mutation_parent_ids,
            round_index=tick_id,
        )
        candidates = list(batch_result.candidates)
        selected_scores = list(batch_result.selected)
        selected_candidates = [item.candidate for item in selected_scores]
        logger.info("[generation-summary] %s", json.dumps(batch_result.generation_stage_metrics, sort_keys=True))
        if not selected_candidates:
            logger.info(
                "No simulation-worthy candidates generated on this tick. top_fail_reasons=%s",
                batch_result.generation_stage_metrics.get("top_fail_reasons", {}),
            )
            return ServiceTickOutcome(status="no_candidates", pending_job_count=0, generated_count=len(candidates))

        recent_fail_rate = self._recent_job_failure_rate(runtime.service_run_id)
        normal_count = min(
            self.config.loop.simulation_batch_size,
            self.config.service.max_pending_jobs,
            len(selected_candidates),
        )
        adjusted_count = normal_count
        if recent_fail_rate > 0.8:
            adjusted_count = max(3, normal_count // 2)  # floor at 3 to avoid throughput collapse
        elif recent_fail_rate > 0.5:
            adjusted_count = max(3, normal_count * 2 // 3)
        selected_scores = selected_scores[:adjusted_count]
        selected_candidates = [item.candidate for item in selected_scores]
        logger.info(
            "Adaptive submission: recent_fail_rate=%.2f, adjusted_count=%s",
            recent_fail_rate,
            adjusted_count,
        )
        batch = self.brain_service.submit_candidates(
            selected_candidates,
            config=self.config,
            environment=self.environment,
            round_index=tick_id,
            batch_size=adjusted_count,
        )
        pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        logger.info(
            "Submitted batch=%s generated=%s submitted=%s pending=%s",
            batch.batch_id,
            len(candidates),
            batch.submitted_count,
            pending_count,
        )
        append_progress_event(
            self.config,
            self.environment,
            event="batch_submitted",
            stage="service-submit",
            status=batch.status,
            tick_id=tick_id,
            round_index=tick_id,
            batch_id=batch.batch_id,
            payload={
                "generated_count": len(candidates),
                "submitted_count": batch.submitted_count,
                "pending_job_count": pending_count,
                "export_path": batch.export_path,
            },
        )
        return ServiceTickOutcome(
            status="running" if pending_count else "idle",
            pending_job_count=pending_count,
            active_batch_id=batch.batch_id,
            generated_count=len(candidates),
            submitted_count=batch.submitted_count,
        )

    def _recover_submitting_batches(
        self,
        *,
        service_name: str,
        run_id: str,
        now: str,
    ) -> tuple[list[str], list[str]]:
        recovered: list[str] = []
        quarantined: list[str] = []
        batches = self.repository.submissions.list_batches_by_status(run_id=run_id, statuses=("submitting",))
        for batch in batches:
            submissions = self.repository.submissions.list_submissions(run_id=run_id, batch_id=batch.batch_id)
            if batch.candidate_count > 0 and len(submissions) >= batch.candidate_count:
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
        return recovered, quarantined

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
