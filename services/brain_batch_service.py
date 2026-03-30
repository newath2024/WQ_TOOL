from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from core.config import AppConfig
from features.registry import build_registry
from generator.engine import AlphaCandidate, AlphaGenerationEngine, GenerationSessionStats
from generator.guided_generator import GuidedGenerator
from memory.pattern_memory import PatternMemoryService, PatternMemorySnapshot
from services.candidate_selection_service import CandidateSelectionService
from services.data_service import CachedResearchContextProvider, resolve_field_registry
from services.evaluation_service import alpha_candidate_from_record
from services.models import BatchPreparationResult, CandidateScore, CommandEnvironment
from storage.models import StageMetricRecord
from storage.repository import SQLiteRepository


class BrainBatchService:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        selection_service: CandidateSelectionService | None = None,
        research_context_provider: CachedResearchContextProvider | None = None,
    ) -> None:
        self.repository = repository
        self.selection_service = selection_service
        self._research_context_provider = research_context_provider
        self._research_context_provider_signature: tuple[bool, int] | None = None

    def prepare_next_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
    ) -> tuple[list[AlphaCandidate], list[CandidateScore]]:
        result = self.prepare_service_batch(
            config=config,
            environment=environment,
            count=count,
            mutation_parent_ids=None,
        )
        return list(result.candidates), list(result.selected)

    def prepare_service_batch(
        self,
        *,
        config: AppConfig,
        environment: CommandEnvironment,
        count: int | None = None,
        mutation_parent_ids: set[str] | None = None,
        round_index: int = 0,
    ) -> BatchPreparationResult:
        provider = self._get_research_context_provider(config)
        cache_result = provider.load(config, environment, stage="brain-sim-data")
        research_context = cache_result.research_context
        active_regime_key = research_context.effective_regime_key or research_context.regime_key
        persistence = provider.persist_metadata(
            self.repository,
            config,
            environment,
            cache_result,
            round_index=round_index,
        )
        resolve_field_registry_started = time.perf_counter()
        field_registry = resolve_field_registry(config, research_context)
        resolve_field_registry_ms = (time.perf_counter() - resolve_field_registry_started) * 1000.0
        registry = build_registry(
            config.generation.allowed_operators,
            operator_catalog_paths=config.generation.operator_catalog_paths,
        )
        snapshot = self.repository.alpha_history.load_snapshot(
            regime_key=active_regime_key,
            region=research_context.region,
            global_regime_key=research_context.global_regime_key,
            parent_pool_size=config.adaptive_generation.parent_pool_size,
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
        generation_count = count or config.loop.generation_batch_size
        mutation_candidates, mutation_stats = self._generate_mutation_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            region_learning_context=research_context.region_learning_context,
            run_id=environment.context.run_id,
            mutation_parent_ids=mutation_parent_ids or set(),
            existing_normalized=existing_normalized,
        )
        fresh_budget = max(0, generation_count - len(mutation_candidates))
        fresh_candidates, fresh_stats = self._generate_fresh_candidates(
            config=config,
            registry=registry,
            field_registry=field_registry,
            snapshot=snapshot,
            case_snapshot=case_snapshot,
            count=fresh_budget,
            existing_normalized=existing_normalized
            | {candidate.normalized_expression for candidate in mutation_candidates},
            memory_service=research_context.memory_service,
            region_learning_context=research_context.region_learning_context,
        )
        candidates = [*mutation_candidates, *fresh_candidates]
        self.repository.save_alpha_candidates(environment.context.run_id, candidates)
        selector = self.selection_service or CandidateSelectionService(
            research_context.memory_service,
            repository=self.repository,
            adaptive_config=config.adaptive_generation,
        )
        selector.configure_runtime(repository=self.repository, adaptive_config=config.adaptive_generation)
        pre_sim_result = selector.run_pre_sim_pipeline(
            candidates,
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
        generation_stats = self._merge_generation_stats(mutation_stats, fresh_stats)
        generation_metrics = generation_stats.to_metrics(
            generated_count=len(candidates),
            selected_for_simulation=len(pre_sim_result.selected),
        )
        generation_metrics.update(
            {
                "load_research_context_ms": round(cache_result.profile.load_research_context_ms, 3),
                "build_field_registry_ms": round(cache_result.profile.build_field_registry_ms, 3),
                "resolve_field_registry_ms": round(resolve_field_registry_ms, 3),
                "prepare_context_ms": round(cache_result.profile.prepare_context_ms, 3),
                "research_context_cache_hit": cache_result.profile.cache_hit,
                "research_context_cache_reason": cache_result.profile.cache_reason,
                "research_context_cache_key": cache_result.profile.cache_key[:12],
                "metadata_dataset_summary_persisted": persistence["dataset_summary_persisted"],
                "metadata_field_catalog_persisted": persistence["field_catalog_persisted"],
                "metadata_run_field_scores_persisted": persistence["run_field_scores_persisted"],
            }
        )
        self.repository.save_stage_metrics(
            [
                StageMetricRecord(
                    run_id=environment.context.run_id,
                    round_index=round_index,
                    stage="generation",
                    metrics_json=json.dumps(generation_metrics, sort_keys=True),
                    created_at=datetime.now(UTC).isoformat(),
                )
            ]
        )
        return BatchPreparationResult(
            candidates=tuple(candidates),
            selected=tuple(pre_sim_result.selected),
            regime_key=active_regime_key,
            generation_stage_metrics=generation_metrics,
        )

    def _generate_fresh_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        case_snapshot,
        count: int,
        existing_normalized: set[str],
        memory_service,
        region_learning_context,
    ) -> tuple[list[AlphaCandidate], GenerationSessionStats]:
        if count <= 0:
            return [], GenerationSessionStats()
        if config.adaptive_generation.enabled:
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=memory_service,
                field_registry=field_registry,
                region_learning_context=region_learning_context,
            )
            candidates = engine.generate(
                count=count,
                snapshot=snapshot,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
            return candidates, engine.last_generation_stats or GenerationSessionStats()
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
        )
        candidates = engine.generate(count=count, existing_normalized=existing_normalized, case_snapshot=case_snapshot)
        return candidates, engine.last_generation_stats or GenerationSessionStats()

    def _generate_mutation_candidates(
        self,
        *,
        config: AppConfig,
        registry,
        field_registry,
        snapshot: PatternMemorySnapshot,
        case_snapshot,
        region_learning_context,
        run_id: str,
        mutation_parent_ids: set[str],
        existing_normalized: set[str],
    ) -> tuple[list[AlphaCandidate], GenerationSessionStats]:
        if not mutation_parent_ids:
            return [], GenerationSessionStats()
        mutation_learning_records = self.repository.list_mutation_outcomes(effective_regime_key=snapshot.regime_key)
        mutation_budget = max(
            1,
            min(
                config.generation.mutation_count,
                len(mutation_parent_ids) * config.loop.max_children_per_parent,
            ),
        )
        if config.adaptive_generation.enabled:
            parent_pool = [parent for parent in snapshot.top_parents if parent.alpha_id in mutation_parent_ids]
            if not parent_pool:
                return [], GenerationSessionStats()
            engine = GuidedGenerator(
                generation_config=config.generation,
                adaptive_config=config.adaptive_generation,
                registry=registry,
                memory_service=self.selection_service.memory_service if self.selection_service else PatternMemoryService(),
                field_registry=field_registry,
                region_learning_context=region_learning_context,
                mutation_learning_records=mutation_learning_records,
            )
            candidates = engine.generate_mutations(
                count=mutation_budget,
                snapshot=snapshot,
                parent_pool=parent_pool,
                existing_normalized=existing_normalized,
                case_snapshot=case_snapshot,
            )
            return candidates, engine.last_generation_stats or GenerationSessionStats()

        parent_refs_map = self.repository.get_parent_refs(run_id)
        parent_records = [
            record
            for record in self.repository.list_alpha_records(run_id)
            if record.alpha_id in mutation_parent_ids
        ]
        parents = [alpha_candidate_from_record(record, parent_refs=parent_refs_map.get(record.alpha_id)) for record in parent_records]
        if not parents:
            return [], GenerationSessionStats()
        engine = AlphaGenerationEngine(
            config=config.generation,
            adaptive_config=config.adaptive_generation,
            registry=registry,
            field_registry=field_registry,
            region_learning_context=region_learning_context,
            mutation_learning_records=mutation_learning_records,
        )
        candidates = engine.generate_mutations(
            parents=parents,
            count=mutation_budget,
            existing_normalized=existing_normalized,
            case_snapshot=case_snapshot,
        )
        return candidates, engine.last_generation_stats or GenerationSessionStats()

    def _get_research_context_provider(self, config: AppConfig) -> CachedResearchContextProvider:
        signature = (
            bool(config.service.research_context_cache_enabled),
            int(config.service.research_context_cache_ttl_seconds),
        )
        if self._research_context_provider is None or self._research_context_provider_signature != signature:
            self._research_context_provider = CachedResearchContextProvider(
                enabled=config.service.research_context_cache_enabled,
                ttl_seconds=config.service.research_context_cache_ttl_seconds,
            )
            self._research_context_provider_signature = signature
        return self._research_context_provider

    def _merge_generation_stats(self, *stats: GenerationSessionStats) -> GenerationSessionStats:
        merged = GenerationSessionStats()
        for item in stats:
            merged.prepare_validation_context_ms += item.prepare_validation_context_ms
            merged.generation_total_ms += item.generation_total_ms
            merged.exploit_phase_ms += item.exploit_phase_ms
            merged.explore_phase_ms += item.explore_phase_ms
            merged.attempt_count += item.attempt_count
            merged.attempt_success_count += item.attempt_success_count
            merged.timeout_stop = merged.timeout_stop or item.timeout_stop
            merged.consecutive_failure_stop = merged.consecutive_failure_stop or item.consecutive_failure_stop
            merged.validation_context_cache_hit = merged.validation_context_cache_hit or item.validation_context_cache_hit
            merged.failure_counts.update(item.failure_counts)
        return merged
