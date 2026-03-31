from __future__ import annotations

import json
from datetime import UTC, datetime
from dataclasses import replace

import yaml

from core.config import AppConfig
from core.logging import get_logger
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine
from generator.guided_generator import GuidedGenerator
from generator.seed_utils import derive_generation_seed
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
from services.progress_log import append_progress_event, resolve_progress_log_path
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
        self.selection_service = selection_service or CandidateSelectionService(
            self.memory_service,
            repository=repository,
        )
        self.brain_service = brain_service
        self.learning_service = BrainLearningService(repository, memory_service=self.memory_service)

    def run(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
    ) -> ClosedLoopRunSummary:
        logger = get_logger(__name__, run_id=environment.context.run_id, stage="closed-loop")
        progress_log_path = resolve_progress_log_path(config, environment)
        progress_log_path_str = str(progress_log_path) if progress_log_path is not None else None
        if progress_log_path_str:
            logger.info("Progress log path=%s", progress_log_path_str)
        research_context = load_research_context(config, environment, stage="closed-loop-data")
        active_regime_key = research_context.effective_regime_key or research_context.regime_key
        self.selection_service.configure_runtime(
            repository=self.repository,
            adaptive_config=config.adaptive_generation,
        )
        field_registry = resolve_field_registry(config, research_context)
        persist_research_metadata(self.repository, config, environment, research_context, round_index=0)
        registry = build_registry(
            config.generation.allowed_operators,
            operator_catalog_paths=config.generation.operator_catalog_paths,
        )
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
        append_progress_event(
            config,
            environment,
            event="closed_loop_run_started",
            stage="closed-loop",
            status="running",
            payload={
                "requested_rounds": config.loop.rounds,
                "generation_batch_size": config.loop.generation_batch_size,
                "simulation_batch_size": config.loop.simulation_batch_size,
                "mutate_top_k": config.loop.mutate_top_k,
            },
        )

        staged_mutations: list[AlphaCandidate] = []
        round_summaries: list[ClosedLoopRoundSummary] = []
        run_status = "completed"
        for round_index in range(1, config.loop.rounds + 1):
            snapshot = self.repository.alpha_history.load_snapshot(
                regime_key=active_regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                parent_pool_size=max(config.adaptive_generation.parent_pool_size, config.loop.mutate_top_k * 2),
                region_learning_config=config.adaptive_generation.region_learning,
                pattern_decay=config.adaptive_generation.pattern_decay,
                prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            )
            case_snapshot = self.repository.alpha_history.load_case_snapshot(
                active_regime_key,
                region=research_context.region,
                global_regime_key=research_context.global_regime_key,
                region_learning_config=config.adaptive_generation.region_learning,
            )
            existing_normalized = self.repository.list_existing_normalized_expressions(environment.context.run_id)
            fresh_budget = max(0, config.loop.generation_batch_size - len(staged_mutations))
            fresh_candidates = self._generate_fresh_candidates(
                config=config,
                registry=registry,
                field_registry=field_registry,
                snapshot=snapshot,
                case_snapshot=case_snapshot,
                region_learning_context=research_context.region_learning_context,
                count=fresh_budget,
                existing_normalized=existing_normalized | {candidate.normalized_expression for candidate in staged_mutations},
                run_id=environment.context.run_id,
                round_index=round_index,
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
                append_progress_event(
                    config,
                    environment,
                    event="closed_loop_round_completed",
                    stage="closed-loop",
                    status=summary.status,
                    round_index=round_index,
                    payload=self._round_summary_payload(summary),
                )
                run_status = "stopped_no_candidates"
                break

            inserted = self.repository.save_alpha_candidates(environment.context.run_id, round_candidates)
            pre_sim_result = self.selection_service.run_pre_sim_pipeline(
                round_candidates,
                snapshot=snapshot,
                field_registry=field_registry,
                batch_size=config.loop.simulation_batch_size,
                min_pattern_support=config.adaptive_generation.min_pattern_support,
                rejection_filters=config.loop.rejection_filters,
                case_snapshot=case_snapshot,
                diversity_config=config.adaptive_generation.diversity,
                run_id=environment.context.run_id,
                round_index=round_index,
                legacy_regime_key=research_context.regime_key,
                global_regime_key=research_context.global_regime_key,
                effective_regime_key=active_regime_key,
            )
            selected = list(pre_sim_result.selected)
            archived = list(pre_sim_result.archived)
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
                selected_parent_results, _, _ = self.selection_service.select_results_for_mutation_with_details(
                    results,
                    candidates_by_id=candidates_by_id,
                    top_k=config.loop.mutate_top_k,
                    diversity_config=config.adaptive_generation.diversity,
                    case_snapshot=case_snapshot,
                    run_id=environment.context.run_id,
                    round_index=round_index,
                )
                selected_parent_ids = {result.candidate_id for result in selected_parent_results}
                self.learning_service.persist_results(
                    config=config,
                    regime_key=active_regime_key,
                    region=research_context.region,
                    global_regime_key=research_context.global_regime_key,
                    market_regime_key=research_context.market_regime_key,
                    effective_regime_key=active_regime_key,
                    regime_label=research_context.regime_label,
                    regime_confidence=research_context.regime_confidence,
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
                    regime_key=active_regime_key,
                    region=research_context.region,
                    global_regime_key=research_context.global_regime_key,
                    region_learning_context=research_context.region_learning_context,
                    case_snapshot=case_snapshot,
                    count=max(1, min(config.generation.mutation_count, len(selected_parent_ids) * config.loop.max_children_per_parent)),
                    existing_normalized=self.repository.list_existing_normalized_expressions(environment.context.run_id),
                    run_id=environment.context.run_id,
                    round_index=round_index,
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
                f"blocked_exact={pre_sim_result.stage_metrics.get('blocked_by_exact_dedup', 0)}",
                f"blocked_near={pre_sim_result.stage_metrics.get('blocked_by_near_duplicate', 0)}",
                f"blocked_cross_run={pre_sim_result.stage_metrics.get('blocked_by_cross_run_dedup', 0)}",
                f"avg_crowding_penalty={float(pre_sim_result.stage_metrics.get('avg_crowding_penalty', 0.0)):.4f}",
            ]
            if batch.export_path:
                notes.append(f"export_path={batch.export_path}")
            summary = ClosedLoopRoundSummary(
                round_index=round_index,
                status=round_status,
                generated_count=len(round_candidates),
                validated_count=int(pre_sim_result.stage_metrics.get("kept_after_dedup", len(round_candidates))),
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
            append_progress_event(
                config,
                environment,
                event="closed_loop_round_completed",
                stage="closed-loop",
                status=summary.status,
                round_index=round_index,
                batch_id=summary.batch_id,
                payload=self._round_summary_payload(summary),
            )
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
        append_progress_event(
            config,
            environment,
            event="closed_loop_run_finished",
            stage="closed-loop",
            status=run_status,
            payload={
                "completed_rounds": len(round_summaries),
                "requested_rounds": config.loop.rounds,
            },
        )
        return ClosedLoopRunSummary(
            run_id=environment.context.run_id,
            backend=config.brain.backend,
            status=run_status,
            rounds=tuple(round_summaries),
            progress_log_path=progress_log_path_str,
        )

    def _generate_fresh_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        case_snapshot,
        region_learning_context,
        count: int,
        existing_normalized: set[str],
        run_id: str,
        round_index: int,
    ) -> list[AlphaCandidate]:
        if count <= 0:
            return []
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="fresh",
        )
        if config.adaptive_generation.enabled:
            engine = GuidedGenerator(
                generation_config=scoped_generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.memory_service,
                field_registry=field_registry,
                region_learning_context=region_learning_context,
            )
            return engine.generate(
                count=count,
                snapshot=snapshot,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
        engine = AlphaGenerationEngine(
            config=scoped_generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
        )
        return engine.generate(count=count, existing_normalized=existing_normalized, case_snapshot=case_snapshot)

    def _generate_mutation_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        selected_parent_ids: set[str],
        candidates_by_id: dict[str, AlphaCandidate],
        regime_key: str,
        region: str,
        global_regime_key: str,
        region_learning_context,
        case_snapshot,
        count: int,
        existing_normalized: set[str],
        run_id: str,
        round_index: int,
    ) -> list[AlphaCandidate]:
        if not selected_parent_ids or count <= 0:
            return []
        mutation_learning_records = self.repository.list_mutation_outcomes(effective_regime_key=regime_key)
        scoped_generation = self._generation_config_for_scope(
            config.generation,
            run_id=run_id,
            round_index=round_index,
            scope="mutation",
        )
        if config.adaptive_generation.enabled:
            snapshot = self.repository.alpha_history.load_snapshot(
                regime_key=regime_key,
                region=region,
                global_regime_key=global_regime_key,
                parent_pool_size=max(config.adaptive_generation.parent_pool_size, len(selected_parent_ids) * 2),
                region_learning_config=config.adaptive_generation.region_learning,
                pattern_decay=config.adaptive_generation.pattern_decay,
                prior_weight=config.adaptive_generation.critic_thresholds.score_prior_weight,
            )
            parent_pool = [parent for parent in snapshot.top_parents if parent.alpha_id in selected_parent_ids]
            if not parent_pool:
                return []
            engine = GuidedGenerator(
                generation_config=scoped_generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.memory_service,
                field_registry=field_registry,
                region_learning_context=region_learning_context,
                mutation_learning_records=mutation_learning_records,
            )
            return engine.generate_mutations(
                count=count,
                snapshot=snapshot,
                parent_pool=parent_pool,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )

        parents = [candidates_by_id[parent_id] for parent_id in selected_parent_ids if parent_id in candidates_by_id]
        if not parents:
            return []
        engine = AlphaGenerationEngine(
            config=scoped_generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            mutation_learning_records=mutation_learning_records,
        )
        return engine.generate_mutations(
            parents=parents,
            count=count,
            existing_normalized=existing_normalized,
            case_snapshot=case_snapshot,
        )

    @staticmethod
    def _generation_config_for_scope(
        generation_config,
        *,
        run_id: str,
        round_index: int,
        scope: str,
    ):
        scoped_seed = derive_generation_seed(
            generation_config.random_seed,
            run_id=run_id,
            round_index=round_index,
            scope=scope,
        )
        return replace(generation_config, random_seed=scoped_seed)

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

    @staticmethod
    def _round_summary_payload(summary: ClosedLoopRoundSummary) -> dict[str, object]:
        return {
            "generated_count": summary.generated_count,
            "validated_count": summary.validated_count,
            "submitted_count": summary.submitted_count,
            "completed_count": summary.completed_count,
            "selected_for_mutation_count": summary.selected_for_mutation_count,
            "mutated_children_count": summary.mutated_children_count,
            "export_path": summary.export_path,
            "notes": list(summary.notes),
        }
