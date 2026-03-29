from __future__ import annotations

from collections import Counter

from data.field_registry import FieldRegistry
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.models import CandidateScore, SimulationResult


class CandidateSelectionService:
    def __init__(self, memory_service: PatternMemoryService | None = None) -> None:
        self.memory_service = memory_service or PatternMemoryService()

    def score_candidates(
        self,
        candidates: list[AlphaCandidate],
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        min_pattern_support: int,
    ) -> list[CandidateScore]:
        scored: list[CandidateScore] = []
        for candidate in candidates:
            family_score, novelty_score, signature, _ = self.memory_service.score_expression(
                candidate.expression,
                snapshot=snapshot,
                min_pattern_support=min_pattern_support,
            )
            field_score = self._average_field_score(candidate, field_registry)
            simplicity_bonus = max(0.0, 1.0 - min(candidate.complexity, 25) / 25.0)
            template_bonus = 0.05 if candidate.template_name else 0.0
            heuristic = 0.55 * field_score + 0.20 * simplicity_bonus + 0.20 * family_score + template_bonus
            scored.append(
                CandidateScore(
                    candidate=candidate,
                    local_heuristic_score=heuristic,
                    novelty_score=novelty_score,
                    family_score=family_score,
                    structural_signature=signature,
                )
            )
        return sorted(
            scored,
            key=lambda item: (
                item.total_score,
                item.novelty_score,
                -item.candidate.complexity,
                item.candidate.alpha_id,
            ),
            reverse=True,
        )

    def select_for_simulation(
        self,
        candidates: list[AlphaCandidate],
        *,
        snapshot: PatternMemorySnapshot,
        field_registry: FieldRegistry,
        batch_size: int,
        min_pattern_support: int,
        rejection_filters: list[str] | None = None,
    ) -> tuple[list[CandidateScore], list[CandidateScore]]:
        scored = self.score_candidates(
            candidates,
            snapshot=snapshot,
            field_registry=field_registry,
            min_pattern_support=min_pattern_support,
        )
        selected: list[CandidateScore] = []
        archived: list[CandidateScore] = []
        selected_signatures: set[str] = set()
        template_counts: Counter[str] = Counter()
        primary_field_counts: Counter[str] = Counter()
        max_per_template = max(1, batch_size // 3)
        max_per_primary_field = max(2, batch_size // 2)
        blocked_tags = set(rejection_filters or [])

        for item in scored:
            fail_tags = set(item.candidate.generation_metadata.get("constraint_tags", []))
            if blocked_tags & fail_tags:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        structural_signature=item.structural_signature,
                        archive_reason="blocked_by_rejection_filter",
                    )
                )
                continue
            family_signature = item.structural_signature.family_signature
            primary_field = item.candidate.fields_used[0] if item.candidate.fields_used else ""
            template_name = item.candidate.template_name or "untyped"
            if family_signature in selected_signatures:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        structural_signature=item.structural_signature,
                        archive_reason="near_duplicate_family",
                    )
                )
                continue
            if template_counts[template_name] >= max_per_template:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        structural_signature=item.structural_signature,
                        archive_reason="template_diversity_cap",
                    )
                )
                continue
            if primary_field and primary_field_counts[primary_field] >= max_per_primary_field:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        structural_signature=item.structural_signature,
                        archive_reason="field_diversity_cap",
                    )
                )
                continue
            selected.append(item)
            selected_signatures.add(family_signature)
            template_counts[template_name] += 1
            if primary_field:
                primary_field_counts[primary_field] += 1
            if len(selected) >= batch_size:
                break

        for item in scored:
            if item not in selected and item not in archived:
                archived.append(
                    CandidateScore(
                        candidate=item.candidate,
                        local_heuristic_score=item.local_heuristic_score,
                        novelty_score=item.novelty_score,
                        family_score=item.family_score,
                        structural_signature=item.structural_signature,
                        archive_reason="fell_below_batch_cutoff",
                    )
                )
        return selected, archived

    def select_results_for_mutation(
        self,
        results: list[SimulationResult],
        *,
        candidates_by_id: dict[str, AlphaCandidate],
        top_k: int,
    ) -> list[SimulationResult]:
        ranked = sorted(
            results,
            key=lambda result: (
                1 if result.status == "completed" else 0,
                1 if result.submission_eligible is True else 0,
                result.metrics.get("fitness") if result.metrics.get("fitness") is not None else float("-inf"),
                result.metrics.get("sharpe") if result.metrics.get("sharpe") is not None else float("-inf"),
                -(result.metrics.get("turnover") or 0.0),
            ),
            reverse=True,
        )
        selected: list[SimulationResult] = []
        family_signatures: set[str] = set()
        for result in ranked:
            candidate = candidates_by_id.get(result.candidate_id)
            if candidate is None:
                continue
            if result.status != "completed":
                continue
            if result.rejection_reason:
                continue
            signature = self.memory_service.extract_signature(candidate.expression)
            if signature.family_signature in family_signatures:
                continue
            selected.append(result)
            family_signatures.add(signature.family_signature)
            if len(selected) >= top_k:
                break
        return selected

    def flag_for_manual_review(self, results: list[SimulationResult]) -> list[SimulationResult]:
        flagged: list[SimulationResult] = []
        for result in results:
            if result.status in {"failed", "rejected"} and not result.rejection_reason:
                flagged.append(result)
        return flagged

    @staticmethod
    def _average_field_score(candidate: AlphaCandidate, field_registry: FieldRegistry) -> float:
        if not candidate.fields_used:
            return 0.0
        scores = [
            field_registry.get(name).field_score
            for name in candidate.fields_used
            if field_registry.contains(name)
        ]
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))
