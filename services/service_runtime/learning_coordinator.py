from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime

from services.brain_learning_service import BrainLearningService
from services.data_service import load_research_context
from services.evaluation_service import alpha_candidate_from_record
from domain.brain import BrainResultRecord
from domain.candidate import AlphaCandidate
from domain.simulation import SimulationResult
from storage.models import (
    ClosedLoopRoundRecord,
    SubmissionBatchRecord,
)

class LearningCoordinator:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def _learn_from_completed_batch(self, *, run_id: str, batch_id: str) -> None:
        results = [
            record
            for record in self.repository.brain_results.list_results(run_id=run_id)
            if record.batch_id == batch_id
        ]
        if not results:
            return
        if all(self._is_neutral_operational_result_record(result) for result in results):
            self._sync_service_round(
                run_id=run_id,
                batch_id=batch_id,
                selected_for_mutation_count=0,
            )
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
        result_rows = self.repository.connection.execute(
            """
            SELECT DISTINCT candidate_id, status, rejection_reason
            FROM brain_results
            WHERE run_id = ? AND batch_id = ?
            """,
            (run_id, batch_id),
        ).fetchall()
        candidate_ids = [
            str(row["candidate_id"])
            for row in result_rows
            if row["candidate_id"] and not self._is_neutral_operational_result_row(row)
        ]
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
    def _is_neutral_operational_result_row(row) -> bool:
        rejection = str(row["rejection_reason"] or "").strip()
        return rejection in BrainLearningService._NEUTRAL_OPERATIONAL_REJECTIONS

    @staticmethod
    def _is_neutral_operational_result_record(record) -> bool:
        rejection = str(record.rejection_reason or "").strip()
        return rejection in BrainLearningService._NEUTRAL_OPERATIONAL_REJECTIONS

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

