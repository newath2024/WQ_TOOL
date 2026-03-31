from __future__ import annotations

import json
from datetime import UTC, datetime

from core.config import AdaptiveGenerationConfig, DiversityThresholdConfig
from data.field_registry import FieldRegistry
from memory.case_memory import CaseMemoryService, CaseMemorySnapshot
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.crowding_service import CrowdingService
from services.duplicate_service import DuplicateService
from services.models import (
    CandidateScore,
    CrowdingScore,
    DedupBatchResult,
    DedupDecision,
    PreSimulationSelectionResult,
    SelectionDecision,
    SimulationResult,
)
from services.selection_service import SelectionService
from storage.models import (
    CrowdingScoreRecord,
    DuplicateDecisionRecord,
    SelectionScoreRecord,
    StageMetricRecord,
)
from storage.repository import SQLiteRepository


class CandidateSelectionService:
    def __init__(
        self,
        memory_service: PatternMemoryService | None = None,
        case_memory_service: CaseMemoryService | None = None,
        *,
        repository: SQLiteRepository | None = None,
        adaptive_config: AdaptiveGenerationConfig | None = None,
    ) -> None:
        self.memory_service = memory_service or PatternMemoryService()
        self.case_memory_service = case_memory_service or CaseMemoryService()
        self.repository = repository
        self.adaptive_config = adaptive_config or AdaptiveGenerationConfig()
        self._build_services()

    def configure_runtime(
        self,
        *,
        repository: SQLiteRepository | None = None,
        adaptive_config: AdaptiveGenerationConfig | None = None,
    ) -> None:
        if repository is not None:
            self.repository = repository
        if adaptive_config is not None:
            self.adaptive_config = adaptive_config
        self._build_services()

    def score_candidates(
        self,
        candidates,
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        min_pattern_support: int,
        case_snapshot: CaseMemorySnapshot | None = None,
        crowding_scores: dict[str, CrowdingScore] | None = None,
        dedup_result: DedupBatchResult | None = None,
    ) -> list[CandidateScore]:
        return self.selection_service.score_pre_sim_candidates(
            list(candidates),
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=min_pattern_support,
            case_snapshot=case_snapshot,
            crowding_scores=crowding_scores,
            dedup_result=dedup_result,
        )

    def run_pre_sim_pipeline(
        self,
        candidates,
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        batch_size: int,
        min_pattern_support: int,
        rejection_filters: list[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_config: DiversityThresholdConfig | None = None,
        run_id: str = "",
        round_index: int = 0,
        legacy_regime_key: str = "",
        global_regime_key: str = "",
        effective_regime_key: str = "",
        persist_metrics: bool = True,
    ) -> PreSimulationSelectionResult:
        candidate_list = list(candidates)
        dedup_result = self._pass_through_dedup_result(candidate_list)
        if self.duplicate_service is not None and run_id:
            dedup_result = self.duplicate_service.filter_pre_sim(
                candidate_list,
                run_id=run_id,
                round_index=round_index,
                legacy_regime_key=legacy_regime_key or snapshot.regime_key,
                global_regime_key=global_regime_key or snapshot.global_regime_key,
            )

        crowding_scores = self._zero_crowding_scores(list(dedup_result.kept_candidates))
        if self.crowding_service is not None:
            crowding_scores = self.crowding_service.score_pre_sim(
                list(dedup_result.kept_candidates),
                run_id=run_id,
                round_index=round_index,
                effective_regime_key=effective_regime_key or snapshot.regime_key,
                case_snapshot=case_snapshot,
            )

        scored = self.score_candidates(
            list(dedup_result.kept_candidates),
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=min_pattern_support,
            case_snapshot=case_snapshot,
            crowding_scores=crowding_scores,
            dedup_result=dedup_result,
        )
        selected, archived, selection_decisions = self.selection_service.select_pre_sim(
            scored,
            field_registry=field_registry,
            batch_size=batch_size,
            rejection_filters=rejection_filters,
            diversity_config=diversity_config,
        )
        kept_count = max(1, len(dedup_result.kept_candidates))
        hard_blocked = sum(1 for score in crowding_scores.values() if score.hard_blocked)
        avg_crowding_penalty = sum(score.total_penalty for score in crowding_scores.values()) / kept_count
        stage_metrics = {
            **dict(dedup_result.stage_metrics),
            "kept_after_hard_dedup": len(dedup_result.kept_candidates),
            "hard_blocked_by_crowding": hard_blocked,
            "avg_crowding_penalty": float(avg_crowding_penalty),
            "selected_for_simulation": len(selected),
            "archived_after_selection": len(archived),
        }
        result = PreSimulationSelectionResult(
            selected=tuple(selected),
            archived=tuple(archived),
            dedup_result=dedup_result,
            crowding_scores=crowding_scores,
            selection_decisions=selection_decisions,
            stage_metrics=stage_metrics,
        )
        if persist_metrics and self.repository is not None and run_id:
            self._persist_pre_sim_pipeline(run_id=run_id, round_index=round_index, result=result)
        return result

    def select_for_simulation(
        self,
        candidates,
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        batch_size: int,
        min_pattern_support: int,
        rejection_filters: list[str] | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
        diversity_config: DiversityThresholdConfig | None = None,
        run_id: str = "",
        round_index: int = 0,
        legacy_regime_key: str = "",
        global_regime_key: str = "",
        effective_regime_key: str = "",
    ) -> tuple[list[CandidateScore], list[CandidateScore]]:
        result = self.run_pre_sim_pipeline(
            candidates,
            snapshot=snapshot,
            field_registry=field_registry,
            batch_size=batch_size,
            min_pattern_support=min_pattern_support,
            rejection_filters=rejection_filters,
            case_snapshot=case_snapshot,
            diversity_config=diversity_config,
            run_id=run_id,
            round_index=round_index,
            legacy_regime_key=legacy_regime_key,
            global_regime_key=global_regime_key,
            effective_regime_key=effective_regime_key,
            persist_metrics=bool(run_id),
        )
        return list(result.selected), list(result.archived)

    def select_results_for_mutation_with_details(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, object],
        top_k: int,
        diversity_config: DiversityThresholdConfig | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
        field_registry: FieldRegistry | None = None,
        run_id: str = "",
        round_index: int = 0,
        persist_metrics: bool = True,
    ) -> tuple[list[SimulationResult], tuple[SelectionDecision, ...], tuple[SelectionDecision, ...]]:
        selected, post_decisions, mutation_decisions = self.selection_service.select_mutation_parents(
            results,
            candidates_by_id=candidates_by_id,
            top_k=top_k,
            diversity_config=diversity_config,
            case_snapshot=case_snapshot,
            field_registry=field_registry,
        )
        if persist_metrics and self.repository is not None and run_id:
            self.persist_selection_decisions(run_id, round_index, [*post_decisions, *mutation_decisions])
        return selected, post_decisions, mutation_decisions

    def select_results_for_mutation(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, object],
        top_k: int,
        diversity_config: DiversityThresholdConfig | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
        field_registry: FieldRegistry | None = None,
    ) -> list[SimulationResult]:
        selected, _, _ = self.select_results_for_mutation_with_details(
            results,
            candidates_by_id=candidates_by_id,
            top_k=top_k,
            diversity_config=diversity_config,
            case_snapshot=case_snapshot,
            field_registry=field_registry,
            persist_metrics=False,
        )
        return selected

    def persist_selection_decisions(
        self,
        run_id: str,
        round_index: int,
        decisions: list[SelectionDecision] | tuple[SelectionDecision, ...],
    ) -> None:
        if self.repository is None or not decisions:
            return
        created_at = datetime.now(UTC).isoformat()
        self.repository.save_selection_scores(
            [
                SelectionScoreRecord(
                    run_id=run_id,
                    round_index=round_index,
                    alpha_id=decision.alpha_id,
                    score_stage=decision.score_stage,
                    composite_score=decision.composite_score,
                    selected=decision.selected,
                    rank=decision.rank,
                    reason_codes_json=json.dumps(list(decision.reason_codes), sort_keys=True),
                    breakdown_json=json.dumps(
                        {
                            "score_stage": decision.breakdown.score_stage if decision.breakdown else decision.score_stage,
                            "composite_score": decision.breakdown.composite_score if decision.breakdown else decision.composite_score,
                            "components": decision.breakdown.components if decision.breakdown else {},
                            "reason_codes": list(decision.breakdown.reason_codes if decision.breakdown else decision.reason_codes),
                        },
                        sort_keys=True,
                    ),
                    created_at=created_at,
                )
                for decision in decisions
            ]
        )

    def flag_for_manual_review(self, results: list[SimulationResult]) -> list[SimulationResult]:
        return [result for result in results if result.status in {"failed", "rejected"} and not result.rejection_reason]

    def _build_services(self) -> None:
        self.selection_service = SelectionService(
            config=self.adaptive_config.selection,
            memory_service=self.memory_service,
            case_memory_service=self.case_memory_service,
        )
        self.duplicate_service = (
            DuplicateService(
                self.repository,
                config=self.adaptive_config.duplicate,
                memory_service=self.memory_service,
            )
            if self.repository is not None
            else None
        )
        self.crowding_service = CrowdingService(
            config=self.adaptive_config.crowding,
            diversity_config=self.adaptive_config.diversity,
        )

    def _persist_pre_sim_pipeline(
        self,
        *,
        run_id: str,
        round_index: int,
        result: PreSimulationSelectionResult,
    ) -> None:
        if self.repository is None:
            return
        created_at = datetime.now(UTC).isoformat()
        self.repository.save_duplicate_decisions(
            [
                DuplicateDecisionRecord(
                    run_id=run_id,
                    round_index=round_index,
                    alpha_id=decision.alpha_id,
                    stage=decision.stage,
                    decision=decision.decision,
                    reason_code=decision.reason_code,
                    matched_run_id=decision.matched_run_id,
                    matched_alpha_id=decision.matched_alpha_id,
                    matched_scope=decision.matched_scope,
                    similarity_score=decision.similarity_score,
                    normalized_match=decision.normalized_match,
                    metrics_json=json.dumps(decision.metrics, sort_keys=True),
                    created_at=created_at,
                )
                for decision in result.dedup_result.decisions
            ]
        )
        self.repository.save_crowding_scores(
            [
                CrowdingScoreRecord(
                    run_id=run_id,
                    round_index=round_index,
                    alpha_id=score.alpha_id,
                    stage=score.stage,
                    total_penalty=score.total_penalty,
                    family_penalty=score.family_penalty,
                    motif_penalty=score.motif_penalty,
                    operator_path_penalty=score.operator_path_penalty,
                    lineage_penalty=score.lineage_penalty,
                    batch_penalty=score.batch_penalty,
                    historical_penalty=score.historical_penalty,
                    hard_blocked=score.hard_blocked,
                    reason_codes_json=json.dumps(list(score.reason_codes), sort_keys=True),
                    metrics_json=json.dumps(score.metrics, sort_keys=True),
                    created_at=created_at,
                )
                for score in result.crowding_scores.values()
            ]
        )
        self.persist_selection_decisions(run_id, round_index, list(result.selection_decisions))
        self.repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id=run_id,
                    round_index=round_index,
                    stage="pre_sim",
                    metrics_json=json.dumps(result.stage_metrics, sort_keys=True),
                    created_at=created_at,
                )
            ]
        )

    def _pass_through_dedup_result(self, candidates: list) -> DedupBatchResult:
        return DedupBatchResult(
            kept_candidates=tuple(candidates),
            blocked_candidates=(),
            decisions=tuple(
                DedupDecision(
                    alpha_id=candidate.alpha_id,
                    normalized_expression=candidate.normalized_expression,
                    stage="pre_sim",
                    decision="kept",
                    reason_code="dedup_not_applied",
                    metrics={"duplicate_risk": 0.0},
                )
                for candidate in candidates
            ),
            stage_metrics={
                "generated": len(candidates),
                "blocked_by_exact_dedup": 0,
                "blocked_by_near_duplicate": 0,
                "blocked_by_cross_run_dedup": 0,
                "kept_after_dedup": len(candidates),
            },
        )

    @staticmethod
    def _zero_crowding_scores(candidates: list) -> dict[str, CrowdingScore]:
        return {
            candidate.alpha_id: CrowdingScore(
                alpha_id=candidate.alpha_id,
                stage="pre_sim",
                total_penalty=0.0,
            )
            for candidate in candidates
        }
