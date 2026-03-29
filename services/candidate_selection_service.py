from __future__ import annotations

from core.config import DiversityThresholdConfig
from data.field_registry import FieldRegistry
from memory.case_memory import CaseMemoryService, CaseMemorySnapshot, ObjectiveVector
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.diversity_manager import DiversityManager
from services.models import CandidateScore, SimulationResult
from services.multi_objective_selection import MultiObjectiveSelectionService, RankedItem


class CandidateSelectionService:
    def __init__(
        self,
        memory_service: PatternMemoryService | None = None,
        case_memory_service: CaseMemoryService | None = None,
    ) -> None:
        self.memory_service = memory_service or PatternMemoryService()
        self.case_memory_service = case_memory_service or CaseMemoryService()
        self.multi_objective = MultiObjectiveSelectionService()

    def score_candidates(
        self,
        candidates,
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        min_pattern_support: int,
        case_snapshot: CaseMemorySnapshot | None = None,
    ) -> list[CandidateScore]:
        ranked_items: list[RankedItem] = []
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
            diversity_score = max(novelty_score, 1.0 - min(1.0, signature.complexity / max(1, 2 * candidate.complexity or 1)))
            objective_vector = self.case_memory_service.predict_objectives(
                generation_metadata=candidate.generation_metadata,
                signature=signature,
                snapshot=case_snapshot,
                novelty_score=novelty_score,
                diversity_score=diversity_score,
            )
            candidate.generation_metadata["region"] = snapshot.region
            candidate.generation_metadata["regime_key"] = snapshot.regime_key
            candidate.generation_metadata["global_regime_key"] = snapshot.global_regime_key
            memory_context = candidate.generation_metadata.get("memory_context")
            if not isinstance(memory_context, dict):
                memory_context = {}
                candidate.generation_metadata["memory_context"] = memory_context
            if snapshot.blend is not None:
                memory_context["pattern_blend"] = snapshot.blend.to_dict()
            if case_snapshot is not None and case_snapshot.blend is not None:
                memory_context["case_blend"] = case_snapshot.blend.to_dict()
            candidate.generation_metadata["selection_objectives"] = objective_vector.to_dict()
            field_score = self._average_field_score(candidate, field_registry)
            heuristic = 0.40 * field_score + 0.30 * family_score + 0.30 * diversity_score
            candidate_score = CandidateScore(
                candidate=candidate,
                objective_vector=objective_vector,
                local_heuristic_score=heuristic,
                novelty_score=novelty_score,
                family_score=family_score,
                diversity_score=diversity_score,
                structural_signature=signature,
                ranking_rationale={
                    "objective_vector": objective_vector.to_dict(),
                    "region": snapshot.region,
                    "regime_key": snapshot.regime_key,
                    "global_regime_key": snapshot.global_regime_key,
                    "pattern_blend": snapshot.blend.to_dict() if snapshot.blend is not None else {},
                    "case_blend": case_snapshot.blend.to_dict() if case_snapshot and case_snapshot.blend is not None else {},
                },
            )
            score_by_id[candidate.alpha_id] = candidate_score
            ranked_items.append(self._candidate_ranked_item(candidate_score, field_registry))
        ordered = self.multi_objective.order(ranked_items)
        return [
            self._with_ranked_rationale(score_by_id[item.item.candidate.alpha_id], item)
            for item in ordered
        ]

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
    ) -> tuple[list[CandidateScore], list[CandidateScore]]:
        scored = self.score_candidates(
            candidates,
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=min_pattern_support,
            case_snapshot=case_snapshot,
        )
        blocked_tags = set(rejection_filters or [])
        diversity_manager = DiversityManager(diversity_config or DiversityThresholdConfig())
        eligible_items: list[RankedItem] = []
        archived: list[CandidateScore] = []
        for item in scored:
            fail_tags = set(item.candidate.generation_metadata.get("constraint_tags", []))
            if blocked_tags & fail_tags:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        objective_vector=item.objective_vector,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        diversity_score=item.diversity_score,
                        structural_signature=item.structural_signature,
                        archive_reason="blocked_by_rejection_filter",
                        ranking_rationale=item.ranking_rationale,
                    )
                )
                continue
            eligible_items.append(self._candidate_ranked_item(item, field_registry))

        selected_ranked, archived_ranked = diversity_manager.select(eligible_items, batch_size=batch_size)
        selected_ids = {item.item.candidate.alpha_id for item in selected_ranked}
        selected_templates = {item.item.candidate.template_name for item in selected_ranked if item.item.candidate.template_name}
        selected = [item for item in scored if item.candidate.alpha_id in selected_ids]
        archived.extend(
            [
                CandidateScore(
                    candidate=item.item.candidate,
                    objective_vector=item.item.objective_vector,
                    local_heuristic_score=item.item.local_heuristic_score,
                    novelty_score=item.item.novelty_score,
                    family_score=item.item.family_score,
                    diversity_score=item.item.diversity_score,
                    structural_signature=item.item.structural_signature,
                    archive_reason=(
                        "template_diversity_cap"
                        if item.item.candidate.template_name and item.item.candidate.template_name in selected_templates
                        else "diversity_cap"
                    ),
                    ranking_rationale=item.item.ranking_rationale,
                )
                for item in archived_ranked
                if item.item.candidate.alpha_id not in selected_ids
            ]
        )
        archived.extend(
            [
                CandidateScore(
                    candidate=item.candidate,
                    objective_vector=item.objective_vector,
                    local_heuristic_score=item.local_heuristic_score,
                    novelty_score=item.novelty_score,
                    family_score=item.family_score,
                    diversity_score=item.diversity_score,
                    structural_signature=item.structural_signature,
                    archive_reason="fell_below_batch_cutoff",
                    ranking_rationale=item.ranking_rationale,
                )
                for item in scored
                if item.candidate.alpha_id not in selected_ids and all(
                    archived_item.candidate.alpha_id != item.candidate.alpha_id for archived_item in archived
                )
            ]
        )
        return selected[:batch_size], archived

    def select_results_for_mutation(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, object],
        top_k: int,
        diversity_config: DiversityThresholdConfig | None = None,
    ) -> list[SimulationResult]:
        ranked_items: list[RankedItem] = []
        for result in results:
            candidate = candidates_by_id.get(result.candidate_id)
            if candidate is None or result.status != "completed":
                continue
            signature = self.memory_service.extract_signature(
                candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
            objective_vector = ObjectiveVector(
                fitness=float(result.metrics.get("fitness") or 0.0),
                sharpe=float(result.metrics.get("sharpe") or 0.0),
                eligibility=1.0 if result.submission_eligible else 0.0,
                robustness=1.0 if not result.rejection_reason else 0.0,
                novelty=float(candidate.generation_metadata.get("selection_objectives", {}).get("novelty", 0.5)),
                diversity=float(candidate.generation_metadata.get("selection_objectives", {}).get("diversity", 0.5)),
                turnover_cost=min(1.0, max(0.0, float(result.metrics.get("turnover") or 0.0) / 2.0)),
                complexity_cost=min(1.0, max(0.0, float(candidate.complexity) / 20.0)),
            )
            ranked_items.append(
                RankedItem(
                    item=result,
                    objective_vector=objective_vector,
                    family_signature=signature.family_signature,
                    primary_field_category=(candidate.generation_metadata.get("field_families") or ["other"])[0],
                    horizon_bucket=signature.horizon_bucket,
                    operator_path_key=">".join(signature.operator_path[:4]),
                    diversity_score=objective_vector.diversity,
                    exploration_candidate=str(candidate.generation_mode).startswith("novelty"),
                )
            )
        ordered = self.multi_objective.order(ranked_items)
        selected_ranked, _ = DiversityManager(diversity_config or DiversityThresholdConfig()).select(ordered, batch_size=top_k)
        return [item.item for item in selected_ranked]

    def flag_for_manual_review(self, results: list[SimulationResult]) -> list[SimulationResult]:
        return [result for result in results if result.status in {"failed", "rejected"} and not result.rejection_reason]

    def _candidate_ranked_item(self, item: CandidateScore, field_registry: FieldRegistry) -> RankedItem[CandidateScore]:
        primary_field = item.candidate.fields_used[0] if item.candidate.fields_used else ""
        primary_field_category = field_registry.get(primary_field).category if primary_field and field_registry.contains(primary_field) else "other"
        operator_path = item.candidate.generation_metadata.get("operator_path") or list(item.structural_signature.operator_path)
        return RankedItem(
            item=item,
            objective_vector=item.objective_vector,
            family_signature=item.structural_signature.family_signature,
            primary_field_category=primary_field_category,
            horizon_bucket=item.structural_signature.horizon_bucket,
            operator_path_key=">".join(operator_path[:4]) if operator_path else "none",
            diversity_score=item.diversity_score,
            exploration_candidate=(
                "novelty" in str(item.candidate.generation_mode)
                or str(item.candidate.generation_metadata.get("mutation_mode") or "") == "novelty"
            ),
            rationale=item.ranking_rationale,
        )

    def _with_ranked_rationale(self, score: CandidateScore, ranked: RankedItem[CandidateScore]) -> CandidateScore:
        return CandidateScore(
            candidate=score.candidate,
            objective_vector=score.objective_vector,
            local_heuristic_score=score.local_heuristic_score,
            novelty_score=score.novelty_score,
            family_score=score.family_score,
            diversity_score=score.diversity_score,
            structural_signature=score.structural_signature,
            archive_reason=score.archive_reason,
            ranking_rationale=ranked.rationale,
        )

    @staticmethod
    def _average_field_score(candidate, field_registry: FieldRegistry) -> float:
        if not candidate.fields_used:
            return 0.0
        scores = [
            field_registry.get(name).field_score
            for name in candidate.fields_used
            if field_registry.contains(name)
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0
