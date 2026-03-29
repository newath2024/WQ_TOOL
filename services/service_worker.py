from __future__ import annotations

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
        self.selection_service = selection_service or CandidateSelectionService()
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
        if runtime.cooldown_until and runtime.cooldown_until > now:
            logger.info("Service is cooling down until %s", runtime.cooldown_until)
            return ServiceTickOutcome(
                status="cooldown",
                pending_job_count=len(self.repository.submissions.list_pending_submissions(runtime.service_run_id)),
                active_batch_id=runtime.active_batch_id,
                cooldown_until=runtime.cooldown_until,
            )

        session_state = self.session_manager.ensure_session(runtime=runtime)
        if session_state.status == "waiting_persona":
            sent, notification_at = self.notification_manager.notify_persona_required(
                runtime=runtime,
                persona_url=str(session_state.persona_url or ""),
            )
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="waiting_persona",
                persona_url=session_state.persona_url,
                persona_wait_started_at=runtime.persona_wait_started_at or now,
                persona_last_notification_at=notification_at if sent else runtime.persona_last_notification_at,
                updated_at=now,
            )
            pending_jobs = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
            logger.warning("Persona verification required; pending_jobs=%s", pending_jobs)
            return ServiceTickOutcome(
                status="waiting_persona",
                pending_job_count=pending_jobs,
                active_batch_id=runtime.active_batch_id,
                persona_url=session_state.persona_url,
            )

        if runtime.persona_url or runtime.persona_wait_started_at:
            self.repository.service_runtime.update_state(
                runtime.service_name,
                status="running",
                persona_url=None,
                persona_wait_started_at=None,
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
        candidates, selected, _ = self.batch_service.prepare_service_batch(
            config=self.config,
            environment=self.environment,
            count=self.config.loop.generation_batch_size,
            mutation_parent_ids=mutation_parent_ids,
        )
        selected_candidates = [item.candidate for item in selected]
        if not selected_candidates:
            logger.info("No simulation-worthy candidates generated on this tick.")
            return ServiceTickOutcome(status="no_candidates", pending_job_count=0, generated_count=len(candidates))

        batch = self.brain_service.submit_candidates(
            selected_candidates,
            config=self.config,
            environment=self.environment,
            round_index=tick_id,
            batch_size=min(self.config.loop.simulation_batch_size, self.config.service.max_pending_jobs),
        )
        pending_count = len(self.repository.submissions.list_pending_submissions(runtime.service_run_id))
        logger.info(
            "Submitted batch=%s generated=%s submitted=%s pending=%s",
            batch.batch_id,
            len(candidates),
            batch.submitted_count,
            pending_count,
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
        candidates_by_id = self._load_candidates_by_ids(
            run_id=run_id,
            candidate_ids={result.candidate_id for result in results},
        )
        snapshot = self.repository.alpha_history.load_snapshot(
            regime_key=regime_key,
            parent_pool_size=max(self.config.adaptive_generation.parent_pool_size, self.config.loop.mutate_top_k * 2),
        )
        selected_parent_ids = {
            result.candidate_id
            for result in self.selection_service.select_results_for_mutation(
                [self._simulation_result_from_record(result) for result in results],
                candidates_by_id=candidates_by_id,
                top_k=self.config.loop.mutate_top_k,
            )
        }
        self.learning_service.persist_results(
            config=self.config,
            regime_key=regime_key,
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
        candidates_by_id = self._load_candidates_by_ids(
            run_id=run_id,
            candidate_ids={result.candidate_id for result in latest_results},
        )
        selected = self.selection_service.select_results_for_mutation(
            [self._simulation_result_from_record(result) for result in latest_results],
            candidates_by_id=candidates_by_id,
            top_k=self.config.loop.mutate_top_k,
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
        if run and run.regime_key:
            return run.regime_key
        research_context = load_research_context(
            self.config,
            self.environment,
            stage="service-learning-data",
        )
        return research_context.regime_key

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
