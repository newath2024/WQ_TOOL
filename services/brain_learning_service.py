from __future__ import annotations

from datetime import UTC, datetime

from core.brain_rejections import extract_invalid_field_from_rejection
from evaluation.critic import AlphaDiagnosis, MutationHint
from generator.engine import AlphaCandidate
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.models import SimulationResult
from storage.models import MutationOutcomeRecord
from storage.repository import SQLiteRepository


class BrainLearningService:
    _NEUTRAL_OPERATIONAL_REJECTIONS = frozenset({"poll_timeout_after_downtime"})

    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        memory_service: PatternMemoryService | None = None,
    ) -> None:
        self.repository = repository
        self.memory_service = memory_service or PatternMemoryService()

    def persist_results(
        self,
        *,
        config,
        regime_key: str,
        region: str = "",
        global_regime_key: str = "",
        market_regime_key: str = "",
        effective_regime_key: str = "",
        regime_label: str = "unknown",
        regime_confidence: float = 0.0,
        snapshot: PatternMemorySnapshot,
        candidates_by_id: dict[str, AlphaCandidate],
        results: list[SimulationResult],
        selected_parent_ids: set[str] | None = None,
    ) -> None:
        timestamp = datetime.now(UTC).isoformat()
        entries: list[dict] = []
        selected = selected_parent_ids or set()
        for result in results:
            if self._is_neutral_operational_result(result):
                continue
            candidate = candidates_by_id.get(result.candidate_id)
            if candidate is None:
                continue
            structural_signature = self.memory_service.extract_signature(
                candidate.expression,
                generation_metadata=candidate.generation_metadata,
            )
            gene_ids = [
                observation.pattern_id
                for observation in self.memory_service.build_observations(
                    structural_signature,
                    generation_metadata=candidate.generation_metadata,
                )
                if observation.pattern_kind == "subexpression"
            ]
            diagnosis = self.assess_result(
                candidate=candidate,
                result=result,
                snapshot=snapshot,
                complexity_limit=config.generation.complexity_limit,
            )
            outcome_score = self.memory_service.compute_brain_outcome_score(
                metrics=result.metrics,
                submission_eligible=result.submission_eligible,
                rejection_reason=result.rejection_reason,
                fail_tags=diagnosis.fail_tags,
            )
            passed_filters = result.status == "completed" and not result.rejection_reason
            entries.append(
                {
                    "candidate": candidate,
                    "result": result,
                    "diagnosis": diagnosis,
                    "structural_signature": structural_signature,
                    "gene_ids": gene_ids,
                    "outcome_score": outcome_score,
                    "behavioral_novelty_score": float(
                        candidate.generation_metadata.get("selection_objectives", {}).get("novelty", 0.5)
                    ),
                    "passed_filters": passed_filters,
                    "selected": candidate.alpha_id in selected,
                    "metric_source": result.metric_source,
                }
            )
        if entries:
            self.repository.alpha_history.persist_brain_outcomes(
                run_id=entries[0]["result"].run_id,
                regime_key=regime_key,
                region=region,
                global_regime_key=global_regime_key,
                market_regime_key=market_regime_key,
                effective_regime_key=effective_regime_key or regime_key,
                regime_label=regime_label,
                regime_confidence=regime_confidence,
                entries=entries,
                pattern_decay=config.adaptive_generation.pattern_decay,
                prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
                created_at=timestamp,
            )
            self._persist_mutation_outcomes(
                run_id=entries[0]["result"].run_id,
                effective_regime_key=effective_regime_key or regime_key,
                entries=entries,
                selected_parent_ids=selected,
                created_at=timestamp,
            )

    def assess_result(
        self,
        *,
        candidate: AlphaCandidate,
        result: SimulationResult,
        snapshot: PatternMemorySnapshot,
        complexity_limit: int,
    ) -> AlphaDiagnosis:
        diagnosis = AlphaDiagnosis()
        fitness = result.metrics.get("fitness")
        sharpe = result.metrics.get("sharpe")
        turnover = result.metrics.get("turnover")
        signature = self.memory_service.extract_signature(
            candidate.expression,
            generation_metadata=candidate.generation_metadata,
        )
        family_observation = self.memory_service.build_observations(signature)[0]
        prior_pattern = snapshot.get_pattern(family_observation.pattern_id, scope="blended")

        rejection = str(result.rejection_reason or "").strip()
        if result.status == "rejected" or rejection:
            diagnosis.fail_tags.append("brain_rejected")
            invalid_field = extract_invalid_field_from_rejection(rejection)
            if invalid_field:
                diagnosis.fail_tags.append(f"invalid_field:{invalid_field}")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="remove_invalid_field", reason=f"BRAIN rejected field: {invalid_field}")
                )
            elif "unknown variable" in rejection.lower():
                diagnosis.mutation_hints.append(
                    MutationHint(hint="remove_invalid_field", reason="BRAIN rejected one field in the expression.")
                )
            elif "does not support event inputs" in rejection.lower():
                diagnosis.fail_tags.append("operator_event_type_mismatch")
                diagnosis.mutation_hints.append(
                    MutationHint(hint="fix_event_operator_mismatch", reason="Operator does not support event inputs")
                )
            else:
                diagnosis.mutation_hints.append(
                    MutationHint(hint="simplify_and_retarget", reason="Rejected by BRAIN, simplify and change structure.")
                )
        if fitness is None or fitness < 0:
            diagnosis.fail_tags.append("poor_fitness")
            diagnosis.mutation_hints.append(
                MutationHint(hint="diversify_feature_family", reason="Improve feature family after weak BRAIN fitness.")
            )
        elif fitness >= 1.0:
            diagnosis.success_tags.append("strong_fitness")
        if sharpe is not None and sharpe >= 1.0:
            diagnosis.success_tags.append("strong_sharpe")
        elif sharpe is not None and sharpe < 0:
            diagnosis.fail_tags.append("weak_validation")
        if turnover is not None and turnover > 1.0:
            diagnosis.fail_tags.append("high_turnover")
            diagnosis.mutation_hints.append(
                MutationHint(hint="smoothen_and_slow_down", reason="Reduce turnover before the next BRAIN round.")
            )
        elif turnover is not None and turnover <= 0.7:
            diagnosis.success_tags.append("turnover_acceptable")
        if result.submission_eligible is True:
            diagnosis.success_tags.append("submission_eligible")
        if candidate.complexity >= int(complexity_limit * 0.8):
            diagnosis.fail_tags.append("excessive_complexity")
            diagnosis.mutation_hints.append(
                MutationHint(hint="reduce_complexity", reason="Prefer fewer layers before resubmission.")
            )
        if prior_pattern is not None and prior_pattern.pattern_score <= 0 and (fitness is None or fitness <= 0):
            diagnosis.fail_tags.append("duplicate_family_no_improvement")
            diagnosis.mutation_hints.append(
                MutationHint(hint="change_template_family", reason="Family repeated without improving BRAIN outcomes.")
            )
        diagnosis.fail_tags = list(dict.fromkeys(diagnosis.fail_tags))
        diagnosis.success_tags = list(dict.fromkeys(diagnosis.success_tags))
        deduped_hints: list[MutationHint] = []
        seen_hints: set[str] = set()
        for hint in diagnosis.mutation_hints:
            if hint.hint in seen_hints:
                continue
            deduped_hints.append(hint)
            seen_hints.add(hint.hint)
        diagnosis.mutation_hints = deduped_hints
        return diagnosis

    @classmethod
    def _is_neutral_operational_result(cls, result: SimulationResult) -> bool:
        rejection = str(result.rejection_reason or "").strip()
        return rejection in cls._NEUTRAL_OPERATIONAL_REJECTIONS

    def _persist_mutation_outcomes(
        self,
        *,
        run_id: str,
        effective_regime_key: str,
        entries: list[dict],
        selected_parent_ids: set[str],
        created_at: str,
    ) -> None:
        records: list[MutationOutcomeRecord] = []
        for entry in entries:
            candidate = entry["candidate"]
            parent_refs = candidate.generation_metadata.get("parent_refs") or [
                {"run_id": run_id, "alpha_id": parent_id}
                for parent_id in candidate.parent_ids
            ]
            if not parent_refs:
                continue
            family_signature = str(candidate.generation_metadata.get("family_signature") or "")
            mutation_mode = str(candidate.generation_metadata.get("mutation_mode") or candidate.generation_mode or "")
            child_score = float(entry["outcome_score"])
            for parent_ref in parent_refs:
                parent_run_id = str(parent_ref.get("run_id") or run_id)
                parent_alpha_id = str(parent_ref.get("alpha_id") or "")
                if not parent_alpha_id:
                    continue
                parent_outcome_score = self.repository.alpha_history.get_outcome_score(parent_run_id, parent_alpha_id)
                parent_score = float(parent_outcome_score or 0.0)
                records.append(
                    MutationOutcomeRecord(
                        run_id=run_id,
                        child_alpha_id=candidate.alpha_id,
                        parent_alpha_id=parent_alpha_id,
                        parent_run_id=parent_run_id,
                        mutation_mode=mutation_mode,
                        family_signature=family_signature,
                        effective_regime_key=effective_regime_key,
                        outcome_source=str(entry.get("metric_source") or "external_brain"),
                        parent_post_sim_score=parent_score,
                        child_post_sim_score=child_score,
                        outcome_delta=child_score - parent_score,
                        selected_for_simulation=True,
                        selected_for_mutation=candidate.alpha_id in selected_parent_ids,
                        created_at=created_at,
                    )
                )
        if records:
            self.repository.save_mutation_outcomes(records)
