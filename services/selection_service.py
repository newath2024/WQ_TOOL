from __future__ import annotations

from collections import Counter

from core.config import DiversityThresholdConfig, SelectionConfig
from data.field_registry import FieldRegistry
from generator.engine import AlphaCandidate
from memory.case_memory import CaseMemorySnapshot, CaseMemoryService, ObjectiveVector
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot, StructuralSignature
from services.diversity_manager import DiversityManager
from services.meta_model_service import MetaModelFeatureInput, MetaModelService
from services.models import (
    CandidateScore,
    CrowdingScore,
    DedupBatchResult,
    SelectionBreakdown,
    SelectionDecision,
    SimulationResult,
)
from services.multi_objective_selection import MultiObjectiveSelectionService, RankedItem


class SelectionService:
    def __init__(
        self,
        *,
        config: SelectionConfig,
        memory_service: PatternMemoryService | None = None,
        case_memory_service: CaseMemoryService | None = None,
        meta_model_service: MetaModelService | None = None,
    ) -> None:
        self.config = config
        self.memory_service = memory_service or PatternMemoryService()
        self.case_memory_service = case_memory_service or CaseMemoryService()
        self.meta_model_service = meta_model_service
        self.multi_objective = MultiObjectiveSelectionService()

    def score_pre_sim_candidates(
        self,
        candidates: list[AlphaCandidate],
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        min_pattern_support: int,
        case_snapshot: CaseMemorySnapshot | None = None,
        crowding_scores: dict[str, CrowdingScore] | None = None,
        dedup_result: DedupBatchResult | None = None,
        run_id: str = "",
        round_index: int = 0,
        effective_regime_key: str = "",
    ) -> list[CandidateScore]:
        crowding_scores = dict(crowding_scores or {})
        duplicate_metrics = self._duplicate_metrics_by_id(dedup_result)
        prepared_inputs: list[dict[str, object]] = []
        ranked_items: list[RankedItem[CandidateScore]] = []
        score_by_id: dict[str, CandidateScore] = {}
        field_categories = {name: spec.category for name, spec in field_registry.fields.items()}
        for candidate in candidates:
            family_score, novelty_score, signature, _ = self.memory_service.score_expression(
                candidate.expression,
                snapshot=snapshot,
                min_pattern_support=min_pattern_support,
                generation_metadata=candidate.generation_metadata,
                field_categories=field_categories,
                scope="blended",
            )
            diversity_score = max(
                novelty_score,
                1.0 - min(1.0, signature.complexity / max(1, 2 * candidate.complexity or 1)),
            )
            objective_vector = self.case_memory_service.predict_objectives(
                generation_metadata=candidate.generation_metadata,
                signature=signature,
                snapshot=case_snapshot,
                novelty_score=novelty_score,
                diversity_score=diversity_score,
            )
            duplicate_risk = float(duplicate_metrics.get(candidate.alpha_id, {}).get("duplicate_risk", 0.0) or 0.0)
            crowding_penalty = float(
                crowding_scores.get(
                    candidate.alpha_id,
                    CrowdingScore(alpha_id=candidate.alpha_id, stage="pre_sim", total_penalty=0.0),
                ).total_penalty
            )
            regime_fit = self._regime_fit(candidate)
            field_score = self._average_field_score(candidate, field_registry)
            heuristic_predicted_quality = self._predicted_quality(objective_vector, field_score=field_score)
            family_diversity = self._family_diversity_bonus(signature, case_snapshot)
            exploration_bonus = self._exploration_bonus(candidate)
            complexity_cost = max(
                float(objective_vector.complexity_cost),
                min(1.0, float(signature.complexity) / 20.0),
            )
            prepared_inputs.append(
                {
                    "candidate": candidate,
                    "family_score": float(family_score),
                    "novelty_score": float(novelty_score),
                    "signature": signature,
                    "diversity_score": float(diversity_score),
                    "objective_vector": objective_vector,
                    "duplicate_risk": float(duplicate_risk),
                    "crowding_penalty": float(crowding_penalty),
                    "regime_fit": float(regime_fit),
                    "field_score": float(field_score),
                    "heuristic_predicted_quality": float(heuristic_predicted_quality),
                    "family_diversity": float(family_diversity),
                    "exploration_bonus": float(exploration_bonus),
                    "complexity_cost": float(complexity_cost),
                    "meta_model_input": MetaModelService.feature_input_from_candidate(
                        candidate=candidate,
                        structural_signature=signature,
                        effective_regime_key=effective_regime_key or snapshot.regime_key,
                        field_score=field_score,
                        novelty_score=novelty_score,
                        family_diversity=family_diversity,
                        duplicate_risk=duplicate_risk,
                        crowding_penalty=crowding_penalty,
                        regime_fit=regime_fit,
                        heuristic_predicted_quality=heuristic_predicted_quality,
                    ),
                }
            )
        meta_predictions = self._meta_model_predictions(
            feature_inputs=[
                item["meta_model_input"]
                for item in prepared_inputs
                if isinstance(item.get("meta_model_input"), MetaModelFeatureInput)
            ],
            run_id=run_id,
            round_index=round_index,
            effective_regime_key=effective_regime_key or snapshot.regime_key,
        )
        for item in prepared_inputs:
            candidate = item["candidate"]
            assert isinstance(candidate, AlphaCandidate)
            objective_vector = item["objective_vector"]
            assert isinstance(objective_vector, ObjectiveVector)
            signature = item["signature"]
            assert isinstance(signature, StructuralSignature)
            heuristic_predicted_quality = float(item["heuristic_predicted_quality"])
            meta_prediction = meta_predictions.get(candidate.alpha_id)
            predicted_quality = (
                float(meta_prediction.blended_predicted_quality)
                if meta_prediction is not None
                else heuristic_predicted_quality
            )
            weights = self.config.pre_sim
            composite_score = (
                weights.predicted_quality * predicted_quality
                + weights.novelty * float(item["novelty_score"])
                + weights.family_diversity * float(item["family_diversity"])
                + weights.regime_fit * float(item["regime_fit"])
                + weights.exploration_bonus * float(item["exploration_bonus"])
                - weights.duplicate_risk * float(item["duplicate_risk"])
                - weights.crowding_penalty * float(item["crowding_penalty"])
                - weights.complexity_cost * float(item["complexity_cost"])
            )
            breakdown = SelectionBreakdown(
                score_stage="pre_sim",
                composite_score=float(composite_score),
                components={
                    "predicted_quality": float(predicted_quality),
                    "heuristic_predicted_quality": float(heuristic_predicted_quality),
                    "ml_positive_outcome_prob": float(meta_prediction.ml_positive_outcome_prob if meta_prediction else 0.0),
                    "blended_predicted_quality": float(predicted_quality),
                    "meta_model_train_rows": int(meta_prediction.train_rows if meta_prediction else 0),
                    "meta_model_positive_rows": int(meta_prediction.positive_rows if meta_prediction else 0),
                    "meta_model_used": 1.0 if (meta_prediction is not None and meta_prediction.used) else 0.0,
                    "field_score": float(item["field_score"]),
                    "novelty": float(item["novelty_score"]),
                    "family_diversity": float(item["family_diversity"]),
                    "regime_fit": float(item["regime_fit"]),
                    "exploration_bonus": float(item["exploration_bonus"]),
                    "duplicate_risk": float(item["duplicate_risk"]),
                    "crowding_penalty": float(item["crowding_penalty"]),
                    "complexity_cost": float(item["complexity_cost"]),
                },
                reason_codes=self._pre_sim_reason_codes(
                    candidate=candidate,
                    duplicate_risk=float(item["duplicate_risk"]),
                    crowding_score=crowding_scores.get(candidate.alpha_id),
                    exploration_bonus=float(item["exploration_bonus"]),
                ),
            )
            memory_context = candidate.generation_metadata.get("memory_context")
            if not isinstance(memory_context, dict):
                memory_context = {}
                candidate.generation_metadata["memory_context"] = memory_context
            if snapshot.blend is not None:
                memory_context["pattern_blend"] = snapshot.blend.to_dict()
            if case_snapshot is not None and case_snapshot.blend is not None:
                memory_context["case_blend"] = case_snapshot.blend.to_dict()
            candidate.generation_metadata["selection_objectives"] = objective_vector.to_dict()
            candidate.generation_metadata["duplicate_risk"] = float(item["duplicate_risk"])
            candidate.generation_metadata["crowding_penalty"] = float(item["crowding_penalty"])
            candidate.generation_metadata["regime_fit"] = float(item["regime_fit"])
            candidate.generation_metadata["heuristic_predicted_quality"] = float(heuristic_predicted_quality)
            candidate.generation_metadata["ml_positive_outcome_prob"] = float(
                meta_prediction.ml_positive_outcome_prob if meta_prediction else 0.0
            )
            candidate.generation_metadata["blended_predicted_quality"] = float(predicted_quality)
            candidate.generation_metadata["meta_model_used"] = bool(meta_prediction.used) if meta_prediction else False
            candidate.generation_metadata["pre_sim_composite_score"] = float(composite_score)
            candidate_score = CandidateScore(
                candidate=candidate,
                objective_vector=objective_vector,
                local_heuristic_score=float(predicted_quality),
                novelty_score=float(item["novelty_score"]),
                family_score=float(item["family_score"]),
                diversity_score=float(item["diversity_score"]),
                duplicate_risk=float(item["duplicate_risk"]),
                crowding_penalty=float(item["crowding_penalty"]),
                regime_fit=float(item["regime_fit"]),
                composite_score=float(composite_score),
                structural_signature=signature,
                reason_codes=breakdown.reason_codes,
                ranking_rationale={
                    "objective_vector": objective_vector.to_dict(),
                    "region": snapshot.region,
                    "regime_key": snapshot.regime_key,
                    "global_regime_key": snapshot.global_regime_key,
                    "pattern_blend": snapshot.blend.to_dict() if snapshot.blend is not None else {},
                    "case_blend": case_snapshot.blend.to_dict() if case_snapshot and case_snapshot.blend is not None else {},
                    "selection_breakdown": {
                        "score_stage": breakdown.score_stage,
                        "composite_score": breakdown.composite_score,
                        "components": dict(breakdown.components),
                        "reason_codes": list(breakdown.reason_codes),
                    },
                },
            )
            score_by_id[candidate.alpha_id] = candidate_score
            ranked_items.append(self._candidate_ranked_item(candidate_score, field_registry))
        ordered = self._finalize_order(ranked_items)
        return [self._with_ranked_rationale(score_by_id[item.item.candidate.alpha_id], item) for item in ordered]

    def select_pre_sim(
        self,
        scored: list[CandidateScore],
        *,
        field_registry: FieldRegistry,
        batch_size: int,
        rejection_filters: list[str] | None = None,
        diversity_config: DiversityThresholdConfig | None = None,
    ) -> tuple[list[CandidateScore], list[CandidateScore], tuple[SelectionDecision, ...]]:
        blocked_tags = set(rejection_filters or [])
        diversity_manager = DiversityManager(diversity_config or DiversityThresholdConfig())
        eligible_items: list[RankedItem[CandidateScore]] = []
        archived: list[CandidateScore] = []
        decisions: list[SelectionDecision] = []
        score_by_id = {item.candidate.alpha_id: item for item in scored}
        for item in scored:
            fail_tags = set(item.candidate.generation_metadata.get("constraint_tags", []))
            if blocked_tags & fail_tags:
                archived_item = self._copy_candidate_score(
                    item,
                    archive_reason="blocked_by_rejection_filter",
                    reason_codes=tuple([*item.reason_codes, "blocked_by_rejection_filter"]),
                )
                archived.append(archived_item)
                decisions.append(self._selection_decision(archived_item, score_stage="pre_sim", selected=False, rank=None))
                continue
            eligible_items.append(self._candidate_ranked_item(item, field_registry))

        ordered = self._finalize_order(eligible_items)
        rank_by_id = {item.item.candidate.alpha_id: index + 1 for index, item in enumerate(ordered)}
        selected_ranked, archived_ranked = diversity_manager.select(ordered, batch_size=batch_size)
        selected_ids = {item.item.candidate.alpha_id for item in selected_ranked}
        selected_templates = {
            item.item.candidate.template_name
            for item in selected_ranked
            if item.item.candidate.template_name
        }
        selected = [score_by_id[item.item.candidate.alpha_id] for item in selected_ranked]
        for score in selected:
            decisions.append(self._selection_decision(score, score_stage="pre_sim", selected=True, rank=rank_by_id.get(score.candidate.alpha_id)))

        for item in archived_ranked:
            candidate_score = score_by_id[item.item.candidate.alpha_id]
            archive_reason = (
                "template_diversity_cap"
                if candidate_score.candidate.template_name and candidate_score.candidate.template_name in selected_templates
                else "diversity_cap"
            )
            archived_item = self._copy_candidate_score(
                candidate_score,
                archive_reason=archive_reason,
                reason_codes=tuple([*candidate_score.reason_codes, archive_reason]),
            )
            archived.append(archived_item)
            decisions.append(
                self._selection_decision(
                    archived_item,
                    score_stage="pre_sim",
                    selected=False,
                    rank=rank_by_id.get(candidate_score.candidate.alpha_id),
                )
            )

        for item in scored:
            if item.candidate.alpha_id in selected_ids:
                continue
            if any(archived_item.candidate.alpha_id == item.candidate.alpha_id for archived_item in archived):
                continue
            archived_item = self._copy_candidate_score(
                item,
                archive_reason="fell_below_batch_cutoff",
                reason_codes=tuple([*item.reason_codes, "fell_below_batch_cutoff"]),
            )
            archived.append(archived_item)
            decisions.append(
                self._selection_decision(
                    archived_item,
                    score_stage="pre_sim",
                    selected=False,
                    rank=rank_by_id.get(item.candidate.alpha_id),
                )
            )

        return selected[:batch_size], archived, tuple(decisions)

    def score_post_sim(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, AlphaCandidate],
        case_snapshot: CaseMemorySnapshot | None = None,
        field_registry: FieldRegistry | None = None,
    ) -> tuple[dict[str, SelectionBreakdown], dict[str, RankedItem[SimulationResult]]]:
        ranked_items: list[RankedItem[SimulationResult]] = []
        breakdowns: dict[str, SelectionBreakdown] = {}
        for result in results:
            candidate = candidates_by_id.get(result.candidate_id)
            if candidate is None or result.status != "completed":
                continue
            signature = self.memory_service.extract_signature(
                candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
            objective_vector = self._result_objective_vector(result=result, candidate=candidate)
            performance_quality = self._performance_quality(objective_vector)
            robustness = float(objective_vector.robustness)
            regime_fit = self._regime_fit(candidate)
            family_diversity_bonus = self._family_diversity_bonus(signature, case_snapshot)
            turnover_margin_cost = self._turnover_margin_cost(result=result, objective_vector=objective_vector)
            crowding_penalty = float(candidate.generation_metadata.get("crowding_penalty") or 0.0)
            duplicate_penalty = float(candidate.generation_metadata.get("duplicate_risk") or 0.0)
            weights = self.config.post_sim
            composite_score = (
                weights.performance_quality * performance_quality
                + weights.robustness * robustness
                + weights.regime_fit * regime_fit
                + weights.family_diversity_bonus * family_diversity_bonus
                - weights.turnover_margin_cost * turnover_margin_cost
                - weights.crowding_penalty * crowding_penalty
                - weights.duplicate_penalty * duplicate_penalty
            )
            breakdown = SelectionBreakdown(
                score_stage="post_sim",
                composite_score=float(composite_score),
                components={
                    "performance_quality": float(performance_quality),
                    "robustness": float(robustness),
                    "regime_fit": float(regime_fit),
                    "family_diversity_bonus": float(family_diversity_bonus),
                    "turnover_margin_cost": float(turnover_margin_cost),
                    "crowding_penalty": float(crowding_penalty),
                    "duplicate_penalty": float(duplicate_penalty),
                },
                reason_codes=self._post_sim_reason_codes(result=result, candidate=candidate),
            )
            breakdowns[result.candidate_id] = breakdown
            ranked_items.append(
                RankedItem(
                    item=result,
                    objective_vector=objective_vector,
                    family_signature=signature.family_signature,
                    primary_field_category=self._primary_field_category(candidate, field_registry=field_registry),
                    horizon_bucket=signature.horizon_bucket,
                    operator_path_key=">".join(signature.operator_path[:4]) if signature.operator_path else "none",
                    diversity_score=objective_vector.diversity,
                    exploration_candidate=self._exploration_bonus(candidate) > 0.0,
                    rationale={
                        "selection_breakdown": {
                            "score_stage": breakdown.score_stage,
                            "composite_score": breakdown.composite_score,
                            "components": dict(breakdown.components),
                            "reason_codes": list(breakdown.reason_codes),
                        }
                    },
                )
            )
        ordered = self._finalize_order(
            ranked_items,
            item_score=lambda item: breakdowns[item.item.candidate_id].composite_score,
        )
        return breakdowns, {item.item.candidate_id: item for item in ordered}

    def select_mutation_parents(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, AlphaCandidate],
        top_k: int,
        diversity_config: DiversityThresholdConfig | None = None,
        case_snapshot: CaseMemorySnapshot | None = None,
        mutation_learnability_by_id: dict[str, float] | None = None,
        field_registry: FieldRegistry | None = None,
    ) -> tuple[list[SimulationResult], tuple[SelectionDecision, ...], tuple[SelectionDecision, ...]]:
        diversity_manager = DiversityManager(diversity_config or DiversityThresholdConfig())
        post_breakdowns, post_ranked = self.score_post_sim(
            results,
            candidates_by_id=candidates_by_id,
            case_snapshot=case_snapshot,
            field_registry=field_registry,
        )
        ordered_post = list(post_ranked.values())
        post_decisions: list[SelectionDecision] = []
        mutation_ranked: list[RankedItem[SimulationResult]] = []
        mutation_breakdowns: dict[str, SelectionBreakdown] = {}
        lineage_counts = Counter(
            str(candidate.generation_metadata.get("lineage_branch_key") or "")
            for result in results
            if result.status == "completed"
            for candidate in [candidates_by_id.get(result.candidate_id)]
            if candidate is not None
        )

        for index, item in enumerate(ordered_post, start=1):
            breakdown = post_breakdowns[item.item.candidate_id]
            post_decisions.append(
                SelectionDecision(
                    alpha_id=item.item.candidate_id,
                    score_stage="post_sim",
                    composite_score=breakdown.composite_score,
                    selected=False,
                    rank=index,
                    reason_codes=breakdown.reason_codes,
                    breakdown=breakdown,
                )
            )
            candidate = candidates_by_id[item.item.candidate_id]
            signature = self.memory_service.extract_signature(
                candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
            family_bonus = self._family_diversity_bonus(signature, case_snapshot)
            lineage_bonus = self._lineage_diversity_bonus(candidate, lineage_counts)
            mutation_learnability_bonus = float((mutation_learnability_by_id or {}).get(candidate.alpha_id, 0.0))
            weights = self.config.mutation_parent
            composite_score = (
                weights.post_sim_score * breakdown.composite_score
                + weights.family_diversification_bonus * family_bonus
                + weights.lineage_diversity_bonus * lineage_bonus
                + weights.mutation_learnability_bonus * mutation_learnability_bonus
            )
            mutation_breakdown = SelectionBreakdown(
                score_stage="mutation_parent",
                composite_score=float(composite_score),
                components={
                    "post_sim_score": float(breakdown.composite_score),
                    "family_diversification_bonus": float(family_bonus),
                    "lineage_diversity_bonus": float(lineage_bonus),
                    "mutation_learnability_bonus": float(mutation_learnability_bonus),
                },
                reason_codes=self._mutation_reason_codes(candidate, mutation_learnability_bonus),
            )
            mutation_breakdowns[candidate.alpha_id] = mutation_breakdown
            mutation_ranked.append(
                RankedItem(
                    item=item.item,
                    objective_vector=item.objective_vector,
                    family_signature=item.family_signature,
                    primary_field_category=item.primary_field_category,
                    horizon_bucket=item.horizon_bucket,
                    operator_path_key=item.operator_path_key,
                    diversity_score=item.diversity_score,
                    exploration_candidate=item.exploration_candidate,
                    rationale={
                        **dict(item.rationale),
                        "selection_breakdown": {
                            "score_stage": mutation_breakdown.score_stage,
                            "composite_score": mutation_breakdown.composite_score,
                            "components": dict(mutation_breakdown.components),
                            "reason_codes": list(mutation_breakdown.reason_codes),
                        },
                    },
                )
            )

        ordered_mutation = self._finalize_order(
            mutation_ranked,
            item_score=lambda item: mutation_breakdowns[item.item.candidate_id].composite_score,
        )
        selected_ranked, _ = diversity_manager.select(ordered_mutation, batch_size=top_k)
        selected_ids = {item.item.candidate_id for item in selected_ranked}
        mutation_decisions: list[SelectionDecision] = []
        for index, item in enumerate(ordered_mutation, start=1):
            breakdown = mutation_breakdowns[item.item.candidate_id]
            mutation_decisions.append(
                SelectionDecision(
                    alpha_id=item.item.candidate_id,
                    score_stage="mutation_parent",
                    composite_score=breakdown.composite_score,
                    selected=item.item.candidate_id in selected_ids,
                    rank=index,
                    reason_codes=breakdown.reason_codes,
                    breakdown=breakdown,
                )
            )
        return [item.item for item in selected_ranked], tuple(post_decisions), tuple(mutation_decisions)

    def _candidate_ranked_item(self, item: CandidateScore, field_registry: FieldRegistry) -> RankedItem[CandidateScore]:
        return RankedItem(
            item=item,
            objective_vector=item.objective_vector,
            family_signature=item.structural_signature.family_signature,
            primary_field_category=self._primary_field_category(item.candidate, field_registry),
            horizon_bucket=item.structural_signature.horizon_bucket,
            operator_path_key=self._operator_path_key(item.structural_signature),
            diversity_score=item.diversity_score,
            exploration_candidate=self._exploration_bonus(item.candidate) > 0.0,
            rationale=item.ranking_rationale,
        )

    def _meta_model_predictions(
        self,
        *,
        feature_inputs: list[MetaModelFeatureInput],
        run_id: str,
        round_index: int,
        effective_regime_key: str,
    ) -> dict[str, object]:
        if self.meta_model_service is None or not feature_inputs:
            return {}
        return self.meta_model_service.score_candidates(
            run_id=run_id,
            round_index=round_index,
            effective_regime_key=effective_regime_key,
            feature_inputs=feature_inputs,
        )

    def _with_ranked_rationale(self, score: CandidateScore, ranked: RankedItem[CandidateScore]) -> CandidateScore:
        rationale = {
            **score.ranking_rationale,
            **dict(ranked.rationale),
            "composite_score": score.composite_score,
        }
        return CandidateScore(
            candidate=score.candidate,
            objective_vector=score.objective_vector,
            local_heuristic_score=score.local_heuristic_score,
            novelty_score=score.novelty_score,
            family_score=score.family_score,
            diversity_score=score.diversity_score,
            duplicate_risk=score.duplicate_risk,
            crowding_penalty=score.crowding_penalty,
            regime_fit=score.regime_fit,
            composite_score=score.composite_score,
            structural_signature=score.structural_signature,
            archive_reason=score.archive_reason,
            reason_codes=score.reason_codes,
            ranking_rationale=rationale,
        )

    def _copy_candidate_score(
        self,
        score: CandidateScore,
        *,
        archive_reason: str,
        reason_codes: tuple[str, ...],
    ) -> CandidateScore:
        rationale = dict(score.ranking_rationale)
        rationale["archive_reason"] = archive_reason
        return CandidateScore(
            candidate=score.candidate,
            objective_vector=score.objective_vector,
            local_heuristic_score=score.local_heuristic_score,
            novelty_score=score.novelty_score,
            family_score=score.family_score,
            diversity_score=score.diversity_score,
            duplicate_risk=score.duplicate_risk,
            crowding_penalty=score.crowding_penalty,
            regime_fit=score.regime_fit,
            composite_score=score.composite_score,
            structural_signature=score.structural_signature,
            archive_reason=archive_reason,
            reason_codes=reason_codes,
            ranking_rationale=rationale,
        )

    def _selection_decision(
        self,
        score: CandidateScore,
        *,
        score_stage: str,
        selected: bool,
        rank: int | None,
    ) -> SelectionDecision:
        payload = dict(score.ranking_rationale.get("selection_breakdown") or {})
        breakdown = SelectionBreakdown(
            score_stage=str(payload.get("score_stage") or score_stage),
            composite_score=float(payload.get("composite_score") or score.composite_score or 0.0),
            components={key: float(value) for key, value in dict(payload.get("components") or {}).items()},
            reason_codes=tuple(payload.get("reason_codes") or score.reason_codes),
        )
        return SelectionDecision(
            alpha_id=score.candidate.alpha_id,
            score_stage=score_stage,
            composite_score=float(score.composite_score or 0.0),
            selected=selected,
            rank=rank,
            reason_codes=tuple(score.reason_codes),
            breakdown=breakdown,
        )

    def _finalize_order(
        self,
        ranked_items: list[RankedItem],
        *,
        item_score=None,
    ) -> list[RankedItem]:
        ordered = self.multi_objective.order(ranked_items)
        score_fn = item_score or (lambda item: float(getattr(item.item, "composite_score", 0.0) or 0.0))
        return sorted(
            ordered,
            key=lambda item: (
                -float(score_fn(item)),
                item.pareto_rank,
                -float(item.crowding_distance),
                item.family_signature,
            ),
        )

    def _predicted_quality(self, objective_vector: ObjectiveVector, *, field_score: float) -> float:
        return float(
            0.20 * field_score
            + 0.30 * objective_vector.fitness
            + 0.20 * objective_vector.sharpe
            + 0.15 * objective_vector.eligibility
            + 0.15 * objective_vector.robustness
        )

    def _performance_quality(self, objective_vector: ObjectiveVector) -> float:
        return float(
            0.55 * objective_vector.fitness
            + 0.30 * objective_vector.sharpe
            + 0.15 * objective_vector.eligibility
        )

    def _turnover_margin_cost(self, *, result: SimulationResult, objective_vector: ObjectiveVector) -> float:
        turnover_cost = float(objective_vector.turnover_cost)
        margin = float(result.metrics.get("margin") or 0.0)
        margin_penalty = max(0.0, min(1.0, 0.10 - margin)) if margin < 0.10 else 0.0
        return float(min(1.0, 0.75 * turnover_cost + 0.25 * margin_penalty))

    def _result_objective_vector(self, *, result: SimulationResult, candidate: AlphaCandidate) -> ObjectiveVector:
        return ObjectiveVector(
            fitness=float(result.metrics.get("fitness") or 0.0),
            sharpe=float(result.metrics.get("sharpe") or 0.0),
            eligibility=1.0 if result.submission_eligible else 0.0,
            robustness=1.0 if not result.rejection_reason else 0.0,
            novelty=float(candidate.generation_metadata.get("selection_objectives", {}).get("novelty", 0.5)),
            diversity=float(candidate.generation_metadata.get("selection_objectives", {}).get("diversity", 0.5)),
            turnover_cost=min(1.0, max(0.0, float(result.metrics.get("turnover") or 0.0) / 2.0)),
            complexity_cost=min(1.0, max(0.0, float(candidate.complexity) / 20.0)),
        )

    def _duplicate_metrics_by_id(
        self,
        dedup_result: DedupBatchResult | None,
    ) -> dict[str, dict[str, float | int | str | bool]]:
        if dedup_result is None:
            return {}
        return {
            decision.alpha_id: dict(decision.metrics)
            for decision in dedup_result.decisions
            if decision.decision == "kept"
        }

    def _family_diversity_bonus(
        self,
        signature: StructuralSignature,
        case_snapshot: CaseMemorySnapshot | None,
    ) -> float:
        if case_snapshot is None:
            return 0.5
        aggregate = case_snapshot.aggregate_for("family", signature.family_signature, scope="blended")
        if aggregate is None:
            return 1.0
        total = max(1, case_snapshot.sample_count + case_snapshot.global_sample_count)
        return float(max(0.0, 1.0 - min(1.0, aggregate.support / total)))

    def _lineage_diversity_bonus(self, candidate: AlphaCandidate, lineage_counts: Counter[str]) -> float:
        lineage_key = str(candidate.generation_metadata.get("lineage_branch_key") or "")
        if not lineage_key:
            return 0.5
        total = max(1, sum(lineage_counts.values()))
        ratio = lineage_counts.get(lineage_key, 0) / total
        return float(max(0.0, 1.0 - min(1.0, ratio)))

    def _regime_fit(self, candidate: AlphaCandidate) -> float:
        explicit = candidate.generation_metadata.get("regime_fit")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                pass
        if candidate.generation_metadata.get("market_regime_key"):
            return 0.5
        return 0.0

    @staticmethod
    def _exploration_bonus(candidate: AlphaCandidate) -> float:
        mode = str(candidate.generation_mode or "")
        mutation_mode = str(candidate.generation_metadata.get("mutation_mode") or "")
        return 1.0 if "novelty" in mode or mutation_mode == "novelty" or "explore" in mode else 0.0

    @staticmethod
    def _pre_sim_reason_codes(
        *,
        candidate: AlphaCandidate,
        duplicate_risk: float,
        crowding_score: CrowdingScore | None,
        exploration_bonus: float,
    ) -> tuple[str, ...]:
        codes: list[str] = []
        if duplicate_risk >= 0.75:
            codes.append("duplicate_risk_high")
        elif duplicate_risk >= 0.50:
            codes.append("duplicate_risk_medium")
        if crowding_score is not None:
            codes.extend(crowding_score.reason_codes)
        if exploration_bonus > 0:
            codes.append("exploration_candidate")
        if candidate.generation_metadata.get("parent_refs"):
            codes.append("has_lineage")
        return tuple(dict.fromkeys(codes))

    @staticmethod
    def _post_sim_reason_codes(*, result: SimulationResult, candidate: AlphaCandidate) -> tuple[str, ...]:
        codes: list[str] = []
        if result.submission_eligible:
            codes.append("submission_eligible")
        if result.rejection_reason:
            codes.append("brain_rejected")
        if float(result.metrics.get("fitness") or 0.0) > 1.0:
            codes.append("strong_fitness")
        if float(result.metrics.get("sharpe") or 0.0) > 1.0:
            codes.append("strong_sharpe")
        if candidate.generation_metadata.get("duplicate_risk", 0.0) >= 0.5:
            codes.append("duplicate_penalty_applied")
        if candidate.generation_metadata.get("crowding_penalty", 0.0) >= 0.5:
            codes.append("crowding_penalty_applied")
        return tuple(dict.fromkeys(codes))

    @staticmethod
    def _mutation_reason_codes(candidate: AlphaCandidate, mutation_learnability_bonus: float) -> tuple[str, ...]:
        codes: list[str] = []
        if candidate.generation_metadata.get("lineage_branch_key"):
            codes.append("lineage_tracked")
        if candidate.generation_metadata.get("parent_refs"):
            codes.append("mutation_parent")
        if mutation_learnability_bonus > 0:
            codes.append("mutation_outcome_supported")
        return tuple(dict.fromkeys(codes))

    @staticmethod
    def _primary_field_category(candidate: AlphaCandidate, field_registry: FieldRegistry | None) -> str:
        primary_field = candidate.fields_used[0] if candidate.fields_used else ""
        if primary_field and field_registry is not None and field_registry.contains(primary_field):
            return field_registry.get(primary_field).category
        field_families = candidate.generation_metadata.get("field_families") or []
        return str(field_families[0]) if field_families else "other"

    @staticmethod
    def _operator_path_key(signature: StructuralSignature) -> str:
        return ">".join(signature.operator_path[:4]) if signature.operator_path else "none"

    @staticmethod
    def _average_field_score(candidate: AlphaCandidate, field_registry: FieldRegistry) -> float:
        if not candidate.fields_used:
            return 0.0
        scores = [
            field_registry.get(name).field_score
            for name in candidate.fields_used
            if field_registry.contains(name)
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0
