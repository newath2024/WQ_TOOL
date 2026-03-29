from __future__ import annotations

import json
from datetime import UTC, datetime

import yaml

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.brain_learning_service import BrainLearningService
from services.brain_service import BrainService
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import (
    load_research_context,
    persist_research_metadata,
    resolve_field_registry,
)
from services.models import (
    ClosedLoopRoundSummary,
    ClosedLoopRunSummary,
    CommandEnvironment,
    SimulationResult,
)
from storage.models import ClosedLoopRoundRecord, ClosedLoopRunRecord
from storage.repository import SQLiteRepository


class ClosedLoopService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        brain_service: BrainService | None = None,
        selection_service: CandidateSelectionService | None = None,
        memory_service: PatternMemoryService | None = None,
    ) -> None:
        self.repository = repository
        self.memory_service = memory_service or PatternMemoryService()
        self.selection_service = selection_service or CandidateSelectionService(self.memory_service)
        self.brain_service = brain_service
        self.learning_service = BrainLearningService(repository, memory_service=self.memory_service)

    def run(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> ClosedLoopRunSummary:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="closed-loop")
        research_context = load_research_context(config, environment, stage="closed-loop-data")
        field_registry = resolve_field_registry(config, research_context)
        persist_research_metadata(self.repository, config, environment, research_context)
        registry = build_registry(config.generation.allowed_operators)
        brain_service = self.brain_service or BrainService(self.repository, config.brain)

        config_snapshot = yaml.safe_dump(config.to_dict(), sort_keys=False)
        self.repository.brain_results.upsert_closed_loop_run(
            ClosedLoopRunRecord(
                run_id=environment.context.run_id,
                backend=config.brain.backend,
                status="running",
                requested_rounds=config.loop.rounds,
                completed_rounds=0,
                config_snapshot=config_snapshot,
                started_at=datetime.now(UTC).isoformat(),
                finished_at=None,
            )
        )

        staged_mutations: list[AlphaCandidate] = []
        round_summaries: list[ClosedLoopRoundSummary] = []
        run_status = "completed"
        for round_index in range(1, config.loop.rounds + 1):
            snapshot = self.repository.alpha_history.load_snapshot(
                regime_key=research_context.regime_key,
                parent_pool_size=max(config.adaptive_generation.parent_pool_size, config.loop.mutate_top_k * 2),
            )
            existing_normalized = self.repository.list_existing_normalized_expressions(environment.context.run_id)
            fresh_budget = max(0, config.loop.generation_batch_size - len(staged_mutations))
            fresh_candidates = self._generate_fresh_candidates(
                config=config,
                registry=registry,
                field_registry=field_registry,
                snapshot=snapshot,
                count=fresh_budget,
                existing_normalized=existing_normalized | {candidate.normalized_expression for candidate in staged_mutations},
            )
            round_candidates = [*staged_mutations, *fresh_candidates]
            if not round_candidates:
                summary = ClosedLoopRoundSummary(
                    round_index=round_index,
                    status="no_candidates",
                    generated_count=0,
                    validated_count=0,
                    submitted_count=0,
                    completed_count=0,
                    selected_for_mutation_count=0,
                    mutated_children_count=0,
                    notes=("No valid candidates remained after generation and dedup.",),
                )
                round_summaries.append(summary)
                self._persist_round_summary(environment, summary)
                run_status = "stopped_no_candidates"
                break

            inserted = self.repository.save_alpha_candidates(environment.context.run_id, round_candidates)
            selected, archived = self.selection_service.select_for_simulation(
                round_candidates,
                snapshot=snapshot,
                field_registry=field_registry,
                batch_size=config.loop.simulation_batch_size,
                min_pattern_support=config.adaptive_generation.min_pattern_support,
                rejection_filters=config.loop.rejection_filters,
            )
            selected_candidates = [item.candidate for item in selected]
            batch = brain_service.simulate_candidates(
                selected_candidates,
                config=config,
                environment=environment,
                round_index=round_index,
                batch_size=config.loop.simulation_batch_size,
            )
            results = list(batch.results)
            candidates_by_id = {candidate.alpha_id: candidate for candidate in round_candidates}

            if results:
                selected_parent_results = self.selection_service.select_results_for_mutation(
                    results,
                    candidates_by_id=candidates_by_id,
                    top_k=config.loop.mutate_top_k,
                )
                selected_parent_ids = {result.candidate_id for result in selected_parent_results}
                self.learning_service.persist_results(
                    config=config,
                    regime_key=research_context.regime_key,
                    snapshot=snapshot,
                    candidates_by_id=candidates_by_id,
                    results=results,
                    selected_parent_ids=selected_parent_ids,
                )
                staged_mutations = self._generate_mutation_candidates(
                    config=config,
                    registry=registry,
                    field_registry=field_registry,
                    selected_parent_ids=selected_parent_ids,
                    candidates_by_id=candidates_by_id,
                    regime_key=research_context.regime_key,
                    count=max(1, min(config.generation.mutation_count, len(selected_parent_ids) * config.loop.max_children_per_parent)),
                    existing_normalized=self.repository.list_existing_normalized_expressions(environment.context.run_id),
                )
                round_status = "completed"
                selected_for_mutation_count = len(selected_parent_ids)
            else:
                staged_mutations = []
                if batch.status == "manual_pending":
                    round_status = "waiting_manual_results"
                    run_status = "waiting_manual_results"
                else:
                    round_status = "completed_without_results"
                    run_status = "completed_without_results"
                selected_for_mutation_count = 0

            notes = [
                f"inserted={inserted}",
                f"archived={len(archived)}",
            ]
            if batch.export_path:
                notes.append(f"export_path={batch.export_path}")
            summary = ClosedLoopRoundSummary(
                round_index=round_index,
                status=round_status,
                generated_count=len(round_candidates),
                validated_count=len(round_candidates),
                submitted_count=len(selected_candidates),
                completed_count=len(results),
                selected_for_mutation_count=selected_for_mutation_count,
                mutated_children_count=len(staged_mutations),
                batch_id=batch.batch_id,
                export_path=batch.export_path,
                notes=tuple(notes),
            )
            round_summaries.append(summary)
            self._persist_round_summary(environment, summary)
            logger.info(
                "Closed-loop round %s status=%s generated=%s submitted=%s completed=%s next_mutations=%s",
                round_index,
                round_status,
                summary.generated_count,
                summary.submitted_count,
                summary.completed_count,
                summary.mutated_children_count,
            )
            if round_status == "waiting_manual_results":
                break

        self.repository.brain_results.upsert_closed_loop_run(
            ClosedLoopRunRecord(
                run_id=environment.context.run_id,
                backend=config.brain.backend,
                status=run_status,
                requested_rounds=config.loop.rounds,
                completed_rounds=len(round_summaries),
                config_snapshot=config_snapshot,
                started_at=environment.context.started_at,
                finished_at=datetime.now(UTC).isoformat(),
            )
        )
        return ClosedLoopRunSummary(
            run_id=environment.context.run_id,
            backend=config.brain.backend,
            status=run_status,
            rounds=tuple(round_summaries),
        )

    def _generate_fresh_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        count: int,
        existing_normalized: set[str],
    ) -> list[AlphaCandidate]:
        if count <= 0:
            return []
        if config.adaptive_generation.enabled:
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.memory_service,
                field_registry=field_registry,
            )
            return engine.generate(
                count=count,
                snapshot=snapshot,
                existing_normalized=existing_normalized,
            )
        engine = AlphaGenerationEngine(config=config.generation, registry=registry, field_registry=field_registry)
        return engine.generate(count=count, existing_normalized=existing_normalized)

    def _generate_mutation_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        selected_parent_ids: set[str],
        candidates_by_id: dict[str, AlphaCandidate],
        regime_key: str,
        count: int,
        existing_normalized: set[str],
    ) -> list[AlphaCandidate]:
        if not selected_parent_ids or count <= 0:
            return []
        if config.adaptive_generation.enabled:
            snapshot = self.repository.alpha_history.load_snapshot(
                regime_key=regime_key,
                parent_pool_size=max(config.adaptive_generation.parent_pool_size, len(selected_parent_ids) * 2),
            )
            parent_pool = [parent for parent in snapshot.top_parents if parent.alpha_id in selected_parent_ids]
            if not parent_pool:
                return []
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.memory_service,
                field_registry=field_registry,
            )
            return engine.generate_mutations(
                count=count,
                snapshot=snapshot,
                parent_pool=parent_pool,
                existing_normalized=existing_normalized,
            )

        parents = [candidates_by_id[parent_id] for parent_id in selected_parent_ids if parent_id in candidates_by_id]
        if not parents:
            return []
        engine = AlphaGenerationEngine(config=config.generation, registry=registry, field_registry=field_registry)
        return engine.generate_mutations(
            parents=parents,
            count=count,
            existing_normalized=existing_normalized,
        )

    def _persist_round_summary(self, environment: CommandEnvironment, summary: ClosedLoopRoundSummary) -> None:
        payload = {
            "batch_id": summary.batch_id,
            "export_path": summary.export_path,
            "notes": list(summary.notes),
        }
        timestamp = datetime.now(UTC).isoformat()
        self.repository.brain_results.upsert_closed_loop_round(
            ClosedLoopRoundRecord(
                run_id=environment.context.run_id,
                round_index=summary.round_index,
                status=summary.status,
                generated_count=summary.generated_count,
                validated_count=summary.validated_count,
                submitted_count=summary.submitted_count,
                completed_count=summary.completed_count,
                selected_for_mutation_count=summary.selected_for_mutation_count,
                mutated_children_count=summary.mutated_children_count,
                summary_json=json.dumps(payload, sort_keys=True),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
